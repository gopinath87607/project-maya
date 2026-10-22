// src/ds4/ds4_mmq.cu - see ds4_mmq.cuh.  A sorted set's piece runs as windows of at most `rows_cap` activation
// rows: for each expert of the window, the rows of its blob [gate; up] (two products into one [rows][2 n_ff] F32
// buffer) then the clamped SwiGLU (ds4_gguf.cu's formula) and the down product.  The activation rows are gathered
// by the chunk's row map (c.row_tok) into MMQ's block layout by quantize_mmq_q8_1_cuda (Ampere) or
// quantize_mmq_fp4_cuda (Blackwell); the products' write-back goes through an identity row map (the window's rows
// are already in sorted order), and the per-expert row ranges are the set's bounds made window-relative on device.
//
// The kernels are ggml's MXFP4 type: a piece is taken only when all three of its expert matrices are MXFP4, and
// when each plane's start and the slot pitch fall on that type's 32-value blocks (17 bytes a block).  A UD quant
// mixes types inside one blob (gate/up IQ2_XS with an MXFP4 down, say), so its pieces keep the GGUF kernels - an
// MXFP4 kernel over an IQ2_S/Q3_K plane reads past the slot (the plane is 3.4-5.1 MB, the MXFP4 read 6.3 MB).
#include "ds4_mmq.cuh"

#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::ds4::mmq {
namespace {

namespace pm = strata::prefill::mmq;

constexpr int kType = gguf::kTypeMXFP4;   // ggml's MXFP4: 32 values in 17 bytes, [out][in / 32]
constexpr size_t kBlock = 17;             // (that type's block: a plane, the slot pitch and each product's k stride)
constexpr int kBounds = 2048;             // (a window never holds more experts: the model has 384)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds4 mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct State {
    int dev = 0, cap = 0, n_embd = 0, n_ff = 0;
    bool used = false;                         // (the first set that runs: a line on stderr, so a run says whether
                                               // it took this path)
    bool declined = false;                     // (one line the first time a piece is turned away, and why)
    pm::Context* ctx = nullptr;
    float* gu = nullptr;                       // [cap][2 n_ff]: the window's gate product, then the up product
    float* h = nullptr;                        // [cap][n_ff]: the clamped SwiGLU
    void* xq = nullptr;                        // the window's activation rows, MMQ's block layout
    void* hq = nullptr;
    int32_t* iota = nullptr;                   // the identity row map the products write back through (+ the kernels'
                                               // stream-k fixup reads its tiles' ids unmasked: twice the rows)
    int32_t* drel = nullptr;                   // the window's bounds, relative to its first row
};

// dst[i] = the set's bound i as a row of the window (clamped: the window can start or end inside an expert)
__global__ void rel_bounds_kernel(int32_t* __restrict__ dst, const int* __restrict__ src, int n, int base, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    int v = src[i] - base;
    dst[i] = v < 0 ? 0 : (v > rows ? rows : v);
}

// ds4_gguf.cu's swiglu: h = silu(min(g, limit)) * clamp(u, -limit, limit), the F32 ops it uses there
__device__ __forceinline__ float swiglu_f(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}

// gu[row][k] (gate) and gu[row][n_ff + k] (up) -> h[row][k]
__global__ void swiglu_split_kernel(const float* __restrict__ gu, float* __restrict__ h, int rows, int n_ff,
                                    float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * n_ff) return;
    const int64_t r = i / n_ff, k = i - r * n_ff;
    const float* row = gu + (size_t) r * (size_t) (2 * n_ff);
    h[i] = swiglu_f(row[k], row[n_ff + k], limit);
}

// One window: experts [wf, we) over rows [wlo, wlo + rows) of the piece (piece-relative), the output into
// c.y's rows [wlo, wlo + rows)
bool window(State& st, const Call& c, int wf, int we, int wlo, int rows, cudaStream_t stream) {
    const x3::SetArgs& sa = *c.sa;
    const int D = st.n_embd, FF = st.n_ff;
    const int i0 = c.i0, n_e = we - wf;
    const size_t tr = (size_t) (wf - i0) * sa.stride;   // the window's first expert from the piece's first
    const int base = sa.row0 + wlo;                      // its first sorted row

    const int* db = sa.bounds + (wf - i0);
    rel_bounds_kernel<<<1, 512, 0, stream>>>(st.drel, db, n_e, base, rows);
    ck(cudaGetLastError(), "rel bounds");

    int max_rows = 0;
    for (int e = wf; e < we; ++e) {
        const int lo = std::max(c.hb[e] - base, 0), hi = std::min(c.hb[e + 1] - base, rows);
        max_rows = std::max(max_rows, hi - lo);
    }
    if (max_rows <= 0) return true;   // (every expert of the window empty here: no rows to write)

    pm::Product p;
    p.type = kType;
    p.expert_bytes = sa.stride;
    p.n = n_e;
    p.bounds = st.drel;
    p.ids = st.iota;
    p.total_rows = rows;
    p.max_rows = max_rows;

    // the activation rows through the row map, block-quantized (the same rows the F32 kernels would read)
    pm::quantize(c.x, (const int32_t*) (c.row_tok + wlo), st.xq, kType, D, c.ldx, rows, stream);
    p.xq = st.xq;

    // gate and up into one [rows][2 n_ff] row (their own products: the blob's gate rows then its up rows)
    p.w = sa.wbase + tr + sa.lay.trellis[0];
    p.w_rows = FF;
    p.w_cols = D;
    p.dst = st.gu;
    p.ld_dst = 2 * FF;
    st.ctx->run(p, stream);
    p.w = sa.wbase + tr + sa.lay.trellis[1];
    p.dst = st.gu + FF;
    st.ctx->run(p, stream);

    const int64_t hn = (int64_t) rows * FF;
    swiglu_split_kernel<<<(int) ((hn + 255) / 256), 256, 0, stream>>>(st.gu, st.h, rows, FF, sa.limit);
    ck(cudaGetLastError(), "swiglu");

    pm::quantize(st.h, nullptr, st.hq, kType, FF, FF, rows, stream);
    p.w = sa.wbase + tr + sa.lay.trellis[2];
    p.w_rows = D;
    p.w_cols = FF;
    p.xq = st.hq;
    p.dst = c.y + (size_t) wlo * (size_t) c.ldy;
    p.ld_dst = c.ldy;
    st.ctx->run(p, stream);
    return true;
}

}  // namespace

void* create(int cuda_dev, cudaStream_t stream, int rows_cap, int n_embd, int n_ff) {
    if (rows_cap < 64) rows_cap = 64;
    if (cudaSetDevice(cuda_dev) != cudaSuccess) return nullptr;
    if (!pm::supported(kType)) return nullptr;
    auto* st = new State();
    st->dev = cuda_dev;
    st->cap = rows_cap;
    st->n_embd = n_embd;
    st->n_ff = n_ff;
    const size_t xq_bytes = pm::act_bytes(kType, rows_cap, n_embd);
    const size_t hq_bytes = pm::act_bytes(kType, rows_cap, n_ff);
    // (the stream-k fixup kernel reads its tiles' ids up to twice the window's rows, unmasked: the identity map
    // covers that, and the main kernel's own tile slack is the +128)
    const size_t n_iota = (size_t) 2 * rows_cap + 128;
    bool ok = cudaMalloc((void**) &st->gu, (size_t) rows_cap * 2 * n_ff * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void**) &st->h, (size_t) rows_cap * n_ff * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&st->xq, xq_bytes) == cudaSuccess && cudaMalloc(&st->hq, hq_bytes) == cudaSuccess &&
              cudaMalloc((void**) &st->iota, n_iota * sizeof(int32_t)) == cudaSuccess &&
              cudaMalloc((void**) &st->drel, kBounds * sizeof(int32_t)) == cudaSuccess;
    if (ok) {
        st->ctx = new pm::Context();   // (the current device's)
        pm::iota(st->iota, (int) n_iota, stream);
        if (cudaGetLastError() != cudaSuccess) ok = false;
    }
    if (!ok) {
        std::fprintf(stderr, "ds4 mmq: CUDA%d has no room for the expert windows (%zu + %zu MiB of activations); the "
                             "prompt path keeps the GGUF kernels\n",
                     cuda_dev, (xq_bytes + (size_t) rows_cap * 2 * n_ff * 4 + (size_t) rows_cap * n_ff * 4) >> 20,
                     ((size_t) rows_cap * n_ff * 4 + hq_bytes) >> 20);
        destroy(st);
        return nullptr;
    }
    std::fprintf(stderr, "ds4 mmq: CUDA%d expert windows, %d rows, %s activations\n", cuda_dev, rows_cap,
                 pm::act_fp4(kType) ? "FP4" : "q8_1");
    return st;
}

void destroy(void* state) {
    if (state == nullptr) return;
    State* st = (State*) state;
    cudaSetDevice(st->dev);
    delete st->ctx;
    cudaFree(st->gu);
    cudaFree(st->h);
    cudaFree(st->xq);
    cudaFree(st->hq);
    cudaFree(st->iota);
    cudaFree(st->drel);
    delete st;
}

bool run(void* state, const Call& c, cudaStream_t stream) {
    State* st = (State*) state;
    if (st == nullptr || c.sa == nullptr) return false;
    const x3::SetArgs& sa = *c.sa;
    if (sa.n_embd != st->n_embd || sa.n_ff != st->n_ff) return false;
    // All three matrices MXFP4, and every plane and the slot pitch on the type's 17-byte blocks (the kernel's rows
    // are 32-value blocks from the plane's start).  Anything else keeps the GGUF kernels.
    if (sa.lay.fmt != gguf::kFmtMXFP4 || gguf::expert_type(sa.lay, 0) != kType ||
        gguf::expert_type(sa.lay, 1) != kType || gguf::expert_type(sa.lay, 2) != kType || sa.n_embd % 32 != 0 ||
        sa.n_ff % 32 != 0 || sa.stride % kBlock != 0 || sa.lay.trellis[1] % kBlock != 0 ||
        sa.lay.trellis[2] % kBlock != 0) {
        if (!st->declined) {
            st->declined = true;
            std::fprintf(stderr,
                         "ds4 mmq: CUDA%d keeps the GGUF kernels for this piece (expert matrices %s / %s / %s, "
                         "stride %zu, planes %zu / %zu; MXFP4 MMQ takes MXFP4 on all three, on 17-byte blocks)\n",
                         st->dev, gguf::type_name(gguf::expert_type(sa.lay, 0)),
                         gguf::type_name(gguf::expert_type(sa.lay, 1)),
                         gguf::type_name(gguf::expert_type(sa.lay, 2)), sa.stride, sa.lay.trellis[1],
                         sa.lay.trellis[2]);
        }
        return false;
    }

    if (!st->used) {
        st->used = true;
        std::fprintf(stderr, "ds4 mmq: CUDA%d takes the prompt experts (%s activations, stride %zu, piece [%d,%d))\n",
                     st->dev, pm::act_fp4(kType) ? "FP4" : "q8_1", sa.stride, c.i0, c.i1);
    }

    const int i0 = c.i0;
    int wf = i0, wlo = 0;   // the piece's first expert; rows of the piece already done
    while (wf < c.i1) {
        const int e_hi = c.hb[wf + 1] - sa.row0;
        if (wlo >= e_hi) {   // (an expert with no rows here, or one the last window finished)
            ++wf;
            continue;
        }
        int we = wf;
        int rows = std::min(st->cap, e_hi - wlo);   // (an expert wider than a window keeps it to cap rows)
        ++we;
        while (we < c.i1) {
            const int ne = c.hb[we + 1] - c.hb[we];
            if (rows + ne > st->cap) break;
            rows += ne;
            ++we;
        }
        if (!window(*st, c, wf, we, wlo, rows, stream)) return false;
        wlo += rows;
        if (wlo >= e_hi) ++wf;
    }
    return true;
}

}  // namespace strata::ds4::mmq
