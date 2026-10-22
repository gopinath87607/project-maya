// src/ds4/ds4_mmq.cuh - the chunked prompt's routed experts through llama.cpp's MMQ kernels (strata::prefill::mmq,
// built from the pinned ggml-cuda): the weights stay MXFP4 and the activations are rounded to q8_1 (an Ampere card)
// or to MMQ's four-bit blocks (a Blackwell card, its native FP4 product), the products on the tensor cores.
// ds4_gguf.cu's set_gate_up / set_down read every expert's F32 weights; this reads the same packed MXFP4 blocks
// directly.  It only applies to a layer whose gate, up and down matrices are all MXFP4 - a UD quant mixes types
// inside one layer (IQ2_XS gate/up with an MXFP4 down, say), and those keep the GGUF kernels.
// Opt-in (--chunk-mmq, DS4_MMQ=1): the prompt path only, decode and the token kernels untouched.
#pragma once

#include "ds4_gguf.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4::mmq {

namespace x3 = strata::kernels::exl3;

/// One sorted set's piece (run_set's shape): experts [i0, i1) of the set, their blobs `sa->stride` apart from
/// `sa->wbase`, rows [sa->row0, ...) of the sorted set.
struct Call {
    const x3::SetArgs* sa = nullptr;   // wbase (the piece's first expert), stride, trellis[3], bounds (device), row0
    const int* hb = nullptr;             // the set's host bounds (absolute sorted rows; hb[i0] = sa->row0)
    const float* x = nullptr;            // the chunk's F32 rows (c.x)
    int ldx = 0;
    const int* row_tok = nullptr;        // sorted row (from sa->row0) -> x row (c.row_tok + sa->row0)
    float* y = nullptr;                  // the chunk's F32 expert output (c.y; piece-relative rows)
    int ldy = 0;
    int i0 = 0, i1 = 0;                  // the piece's set indices
};

/// The per-device state (the launch context and the scratch the windows take): null when this build, card or the
/// card's free memory cannot run it - the caller keeps ds4_gguf.cu's kernels then.
void* create(int cuda_dev, cudaStream_t stream, int rows_cap, int n_embd, int n_ff);
void destroy(void* state);
/// Run the piece's experts over its rows, window by window (at most `rows_cap` activation rows a product), the
/// results in c.y exactly where set_down leaves them.  False (nothing launched) when the piece is not this path's
/// shape: its three expert matrices not all MXFP4 (a UD quant's layers mix types), a stride or plane not in whole
/// 17-byte blocks, another width.
bool run(void* state, const Call& c, cudaStream_t stream);

}  // namespace strata::ds4::mmq
