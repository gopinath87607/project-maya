// src/ds4/ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
//
// The layer, for one token at position p (DeepSeek's inference/model.py, Block.forward with the lagged pre-mix):
//   [Engram, layers 1 and 14]  h += gate(h, key) * value from 24 n-gram rows of the FP8 tables
//   a_pre, a_post, a_comb = hc_mixes(h);  x = rmsnorm(sum_s pre[s] h[s])   (pre: the previous layer's f_pre)
//   attention(x): q from the low-rank wq_a / wq_b, one shared K = V row per token in a 128-slot window ring, plus
//                 (compress ratio > 0) up to 512 compressed rows the indexer picks; a learned sink per head; the
//                 output's rotary part rotated back, rounded to FP8, through the grouped wo_a and wo_b
//   h = post * attn + comb^T h;  f_* = hc_mixes(h);  x = rmsnorm(sum_s a_pre[s] h[s])
//   moe(x): 6 of 384 routed experts (sqrt-softplus scores, biased top-k) + the shared expert
//   h = post * moe + comb^T h;  pre = f_pre
// and after the last layer logits = head(rmsnorm(sum_s pre[s] h[s])).
#include "ds4_model.hpp"
#include "ds4_engram_io.hpp"

#include "ds4_gguf.cuh"
#include "ds4_gguf_cpu.hpp"
#include "ds4_kernels.cuh"
#if defined(STRATA_DS4_MMQ)
#include "ds4_mmq.cuh"
#endif
#include "ds4_os.hpp"
#include "ds4_pool.hpp"
#include "ds4_vision.cuh"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/sampler.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
#define strtok_r strtok_s
#endif

namespace strata::ds4 {
namespace x3 = strata::kernels::exl3;
namespace kc = strata::kernels::cpu;
namespace gf = strata::kernels::glmf;
namespace gg = strata::ds4::gguf;

#define CK(call)                                                                            \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                       \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

// DS4_DBG_HC: a layer's hyper-connection buffers' NaN counts (a token a count) and, when used > 0, the first
// `used` values of each token's row (a debug run's only)
void dbg_nan(const char* tag, int il, int T, const float* p, size_t stride, int used, cudaStream_t s) {
    static const bool on = std::getenv("DS4_DBG_HC") != nullptr;
    if (!on || p == nullptr || il > 1 || T <= 0 || T > 16 || stride == 0 || stride > 65536) return;
    std::vector<float> b((size_t) T * stride);
    if (cudaStreamSynchronize(s) != cudaSuccess ||
        cudaMemcpy(b.data(), p, b.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
        return;
    const int u = used > 0 ? std::min(used, (int) stride) : 0;
    std::string line = std::string(tag) + " il " + std::to_string(il);
    for (int t = 0; t < T; ++t) {
        int n = 0;
        for (size_t i = 0; i < stride; ++i) n += b[(size_t) t * stride + i] != b[(size_t) t * stride + i];
        line += " t" + std::to_string(t) + ":" + std::to_string(n);
        if (u > 0) {
            char v[32];
            line += " [";
            for (int i = 0; i < u; ++i) {
                const float x = b[(size_t) t * stride + i];
                if (x != x) {
                    line += " nan";
                } else {
                    std::snprintf(v, sizeof v, " %g", (double) x);
                    line += v;
                }
            }
            line += " ]";
        }
    }
    std::fprintf(stderr, "DS4_DBG_HC %s\n", line.c_str());
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores (the first logical CPU of each core), in order; `all` too, every logical CPU of
// the node (its siblings included) - the list a pool reaches into when --cpu-threads asks for more than the cores
std::vector<std::vector<int>> node_cores(std::vector<std::vector<int>>* all = nullptr) {
    std::vector<std::vector<int>> nodes, every;
#if defined(_WIN32)
    nodes = os::node_cores();
    every = nodes;   // (the platform's view has no siblings to add)
#else
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        const std::vector<int> cpus = cpu_list(cl);
        std::vector<int> cores;
        for (int c : cpus) {
            const std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty() || sib[0] == c) cores.push_back(c);
        }
        if (!cores.empty()) {
            nodes.push_back(cores);
            every.push_back(cpus);
        }
    }
#endif
    if (nodes.empty()) {
        std::vector<int> only;
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) only.push_back((int) c);
        nodes.push_back(only);
        every.assign(1, only);
    }
    if (all) *all = every;
    return nodes;
}
// anonymous memory on one NUMA node (node < 0: wherever the first touch puts it), transparent huge pages
void* node_map(size_t bytes, int node, int n_nodes) { return os::anon_alloc(bytes, node, n_nodes); }
void pin_self(int cpu) { os::pin_self(cpu); }
// the RAM this process can still get: MemAvailable, and below it the tightest cgroup v2 limit on the way to the root
// (its limit less what is charged, the page cache counted as free: the kernel takes it back); ~0: not known
uint64_t ram_avail() {
#if defined(_WIN32)
    return os::ram_avail();
#endif
    uint64_t avail = ~uint64_t{0};
    {
        std::istringstream mi(read_file("/proc/meminfo"));
        std::string k;
        uint64_t kb = 0;
        while (mi >> k >> kb) {
            if (k == "MemAvailable:") avail = kb << 10;
            mi.ignore(64, '\n');
        }
    }
    std::string cg = read_file("/proc/self/cgroup");   // "0::/user.slice/..."
    const size_t at = cg.find("0::");
    if (at == std::string::npos) return avail;
    cg = cg.substr(at + 3);
    cg = cg.substr(0, cg.find('\n'));
    for (;;) {
        const std::string dir = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
        const std::string mx = read_file(dir + "/memory.max");
        if (!mx.empty() && mx.compare(0, 3, "max") != 0) {
            const uint64_t lim = std::strtoull(mx.c_str(), nullptr, 10);
            const uint64_t cur = std::strtoull(read_file(dir + "/memory.current").c_str(), nullptr, 10);
            uint64_t file = 0;
            std::istringstream st(read_file(dir + "/memory.stat"));
            std::string k;
            uint64_t v = 0;
            while (st >> k >> v)
                if (k == "file") file = v;
            const uint64_t used = cur > file ? cur - file : 0;
            avail = std::min(avail, lim > used ? lim - used : 0);
        }
        if (cg.empty() || cg == "/") break;
        const size_t sl = cg.rfind('/');
        cg = sl == 0 || sl == std::string::npos ? "/" : cg.substr(0, sl);
    }
    return avail;
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f32_to_bf16(float f) {   // round to nearest even (ggml's ggml_fp32_to_bf16_rne)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) ((u >> 16) | ((u & 0xffffu) ? 1u : 0u));
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// ------------------------------------------------------------------ the pack
struct Tensor {
    std::string dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Pack {
    std::string dir, model_dir;
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::vector<os::fd_t> fds;
    std::unordered_map<std::string, Tensor> t;
    int eg_vocab = 0, eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;
    // format gguf (tools/ds4/pack_gguf.py): the files are a GGUF's shards, every one mapped at load (map()) - the
    // routed experts are always read in place (the OS page cache their RAM tier, the disk the rest)
    bool gguf = false;

    std::vector<std::pair<uint8_t*, size_t>> maps;   // file -> its read-only mapping (map(): the experts in place)
    std::vector<os::fd_t> dfds;   // (gguf) file -> unbuffered (os::open_direct; kBadFd: buffered reads only)
    std::vector<os::fd_t> pfds;   // (gguf) file -> for reads from many threads at once (os::open_par)

    ~Pack() {
        for (auto& m : maps) os::unmap(m.first, m.second);
        for (os::fd_t fd : fds) os::close_fd(fd);
        for (os::fd_t fd : dfds) os::close_fd(fd);
        for (os::fd_t fd : pfds) os::close_fd(fd);
    }
    // file i mapped whole, read-only (once; not thread-safe: called while loading)
    const uint8_t* map(int file) {
        if (maps.size() < fds.size()) maps.resize(fds.size(), {nullptr, 0});
        auto& m = maps[(size_t) file];
        if (m.first == nullptr) {
            const uint64_t size = os::file_size(fds[(size_t) file]);
            void* p = os::map_ro(fds[(size_t) file], size);
            if (p == nullptr) return nullptr;
            m = {(uint8_t*) p, (size_t) size};
        }
        return m.first;
    }

    bool load(const std::string& d, std::string& err) {
        dir = d;
        {
            std::ifstream f(d + "/ds4.txt");
            if (!f) {
                err = d + "/ds4.txt: not found (run tools/ds4/pack.py --model <model>)";
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                kv[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir = str("model_dir");
        std::FILE* f = std::fopen((d + "/tensors.txt").c_str(), "r");
        if (f == nullptr) {
            err = d + "/tensors.txt: not found";
            return false;
        }
        std::vector<char> buf(1 << 16);
        while (std::fgets(buf.data(), (int) buf.size(), f) != nullptr) {
            char* s = buf.data();
            if (s[0] == 'f') {   // file <i> <name>
                char name[4096];
                int i = 0;
                if (std::sscanf(s, "file %d %4095s", &i, name) == 2) {
                    if ((int) files.size() <= i) files.resize((size_t) i + 1);
                    files[(size_t) i] = name;
                }
                continue;
            }
            if (s[0] != 't') continue;
            char* save = nullptr;
            ::strtok_r(s, " \n", &save);
            const char* name = ::strtok_r(nullptr, " \n", &save);
            const char* dt = ::strtok_r(nullptr, " \n", &save);
            const char* fi = ::strtok_r(nullptr, " \n", &save);
            const char* off = ::strtok_r(nullptr, " \n", &save);
            const char* nb = ::strtok_r(nullptr, " \n", &save);
            if (name == nullptr || nb == nullptr) continue;
            Tensor x;
            x.dtype = dt;
            x.file = std::atoi(fi);
            x.off = std::strtoull(off, nullptr, 10);
            x.bytes = std::strtoull(nb, nullptr, 10);
            while (const char* dim = ::strtok_r(nullptr, " \n", &save)) x.shape.push_back(std::atoll(dim));
            t.emplace(name, std::move(x));
        }
        std::fclose(f);
        for (const std::string& fn : files) {
            const std::string p = model_dir + "/" + fn;
            const os::fd_t fd = os::open_ro(p);
            if (fd == os::kBadFd) {
                err = p + ": cannot open";
                return false;
            }
            fds.push_back(fd);
        }
        gguf = str("format") == "gguf";
        if (gguf)
            for (size_t i = 0; i < fds.size(); ++i) {
                if (map((int) i) == nullptr) {
                    err = model_dir + "/" + files[i] + ": cannot map";
                    return false;
                }
                dfds.push_back(os::open_direct(model_dir + "/" + files[i]));
                pfds.push_back(os::open_par(model_dir + "/" + files[i]));
            }
        // the Engram tables
        std::FILE* e = std::fopen((d + "/engram.bin").c_str(), "rb");
        if (e == nullptr) {
            err = d + "/engram.bin: not found";
            return false;
        }
        char magic[4];
        int32_t hdr[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 &&
                  std::fread(hdr, 4, 6, e) == 6;
        if (ok) {
            eg_vocab = hdr[0], eg_layers = hdr[1], eg_ngram = hdr[2], eg_heads = hdr[3], eg_hd = hdr[4], eg_pad = hdr[5];
            const int cols = (eg_ngram - 1) * eg_heads;
            eg_map.resize((size_t) eg_vocab);
            eg_mult.resize((size_t) eg_layers * eg_ngram);
            eg_primes.resize((size_t) eg_layers * cols);
            eg_off.resize((size_t) eg_layers * cols);
            ok = std::fread(eg_map.data(), 4, eg_map.size(), e) == eg_map.size() &&
                 std::fread(eg_mult.data(), 8, eg_mult.size(), e) == eg_mult.size() &&
                 std::fread(eg_primes.data(), 8, eg_primes.size(), e) == eg_primes.size() &&
                 std::fread(eg_off.data(), 8, eg_off.size(), e) == eg_off.size();
        }
        std::fclose(e);
        if (!ok) {
            err = d + "/engram.bin: unreadable";
            return false;
        }
        return true;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    bool read_at(int file, uint64_t off, uint64_t n, void* dst) const {   // (any thread; side by side where pfds is)
        if ((size_t) file < pfds.size() && pfds[(size_t) file] != os::kBadFd)
            return os::pread_par(pfds[(size_t) file], dst, n, off);
        return os::pread_full(fds[(size_t) file], dst, n, off);
    }
    bool read(const Tensor& x, void* dst) const { return read_at(x.file, x.off, x.bytes, dst); }
    const uint8_t* at(int file, uint64_t off) const { return maps[(size_t) file].first + off; }   // (gguf: mapped)
};

// ------------------------------------------------------------------ geometry
struct Geo {
    int vocab = 0, dim = 0, nl = 0, heads = 0, hd = 0, rd = 0, q_lora = 0, o_lora = 0, o_groups = 0, n_exp = 0,
        k_exp = 0, ff = 0, win = 0, hc = 0, sinkhorn = 0, ix_heads = 0, ix_hd = 0, ix_topk = 0, cand_src = -1,
        cand_topk = 0, cand_blk = 0, orig_ctx = 0, eos = 1;
    float eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.0f, lim = 0.0f;
    double rope_theta = 1e4, comp_theta = 1e4, factor = 1.0, beta_fast = 32, beta_slow = 1;
    std::vector<int> ratios, kv_src, ix_src, eg_layers;
    // the DSpark drafter (the checkpoint's mtp.* blocks): a block of [seed, noise x (block - 1)] through n_mtp blocks
    int n_mtp = 0, d_block = 0, d_noise = 0, d_rank = 0, d_n_exp = 0, d_k = 0;
    std::vector<int> d_targets;

    void from(const Pack& p) {
        vocab = (int) p.num("vocab_size", 0), dim = (int) p.num("hidden_size", 0);
        nl = (int) p.num("num_hidden_layers", 0), heads = (int) p.num("num_attention_heads", 0);
        hd = (int) p.num("head_dim", 0), rd = (int) p.num("qk_rope_head_dim", 0);
        q_lora = (int) p.num("q_lora_rank", 0), o_lora = (int) p.num("o_lora_rank", 0);
        o_groups = (int) p.num("o_groups", 0), n_exp = (int) p.num("n_routed_experts", 0);
        k_exp = (int) p.num("num_experts_per_tok", 0), ff = (int) p.num("moe_intermediate_size", 0);
        win = (int) p.num("sliding_window", 0), hc = (int) p.num("hc_mult", 0);
        sinkhorn = (int) p.num("hc_sinkhorn_iters", 0), ix_heads = (int) p.num("index_n_heads", 0);
        ix_hd = (int) p.num("index_head_dim", 0), ix_topk = (int) p.num("index_topk", 0);
        cand_src = (int) p.num("candidate_source_layer_id", -1), cand_topk = (int) p.num("candidate_topk_blocks", 0);
        cand_blk = (int) p.num("candidate_block_size", 0);
        orig_ctx = (int) p.num("rope_scaling.original_max_position_embeddings", 0);
        eos = (int) p.num("eos_token_id", 1);
        eps = (float) p.num("rms_norm_eps", 1e-20), hc_eps = (float) p.num("hc_eps", 1e-6);
        route_scale = (float) p.num("routed_scaling_factor", 1.0), lim = (float) p.num("swiglu_limit", 0.0);
        rope_theta = p.num("rope_theta", 1e4), comp_theta = p.num("compress_rope_theta", 1e4);
        factor = p.num("rope_scaling.factor", 1.0), beta_fast = p.num("rope_scaling.beta_fast", 32);
        beta_slow = p.num("rope_scaling.beta_slow", 1);
        ratios = p.list("compress_ratios"), kv_src = p.list("kv_source_layer_ids");
        ix_src = p.list("index_source_layer_ids"), eg_layers = p.list("engram_layer_ids");
        d_block = (int) p.num("dspark_block_size", 0), d_noise = (int) p.num("dspark_noise_token_id", 0);
        d_rank = (int) p.num("dspark_markov_rank", 0), d_n_exp = (int) p.num("dspark_n_routed_experts", 0);
        d_k = (int) p.num("dspark_num_experts_per_tok", 0), d_targets = p.list("dspark_target_layer_ids");
        // the DSpark blocks: an EXL3 pack's, or a GGUF pack's from DeepSeek's checkpoint (tools/ds4/drafter_pack.py)
        n_mtp = d_block > 0 && (int) ratios.size() > nl &&
                        (p.get("mtp.0.attn.wq_a.trellis") != nullptr || p.get("mtp.0.attn.wq_a.weight") != nullptr)
                    ? (int) ratios.size() - nl : 0;
    }
};

// DeepSeek's precompute_freqs_cis frequencies (YaRN when original_seq_len > 0), in F32 as torch computes them
std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f((size_t) dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = (float) (2 * i) / (float) dim;
        f[(size_t) i] = 1.0f / std::pow((float) base, e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base));   // (M_PI: not MSVC's)
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), (double) dim - 1);
        const float den = (float) std::max(high - low, 1e-3);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[(size_t) i] = f[(size_t) i] / (float) factor * (1.0f - smooth) + f[(size_t) i] * smooth;
        }
    }
    return f;
}

// ------------------------------------------------------------------ devices, layers
// a bump arena for the row-split prefill's replica layers: while d.bump is set, dalloc's VRAM allocations take their
// bytes from here (one cudaMalloc a swap instead of ~200), and the buffers they replace are the arena itself - freed
// by the next swap's off = 0, not by d.allocs
struct Bump {
    uint8_t* base = nullptr;
    size_t bytes = 0, off = 0;
    int owner = -1;   // the device it is carved on (a guard against a stray dalloc on another)
};

struct Dev {
    int id = 0;
    std::string pci;   // its bus id (cudaDeviceGetPCIBusId), so the Monitor joins it with the cards NVML lists
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    size_t free0 = 0, total = 0, weights = 0, experts = 0, caches = 0, scratch = 0;
    // the prompt path's MMQ state (ds4_mmq.cu: the launch context and the windows' scratch), made on the first
    // MXFP4 set --chunk-mmq sends its way; tried = created or refused (the card's free memory), so once
    void* mmq = nullptr;
    bool mmq_tried = false;
    std::vector<void*> allocs;
    // the row-split prefill's replica arenas: while set, dalloc's VRAM allocations are carved from it (the swap's
    // dense weights) and a failed carve (off past bytes) is this device's error, not a cudaMalloc
    Bump* bump = nullptr;
    // the prompt's VRAM: one cudaMalloc of exactly what the chunk buffers below take (the ring included), carved at
    // load.  Between prompts the same bytes are the experts' lend banks (Layer::lend_base).  No layer's expert bank
    // ever overlaps another's: the arena's offsets are laid out once, after place_experts decides who lends.
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0, arena_off = 0;
    // the arena's carve layout: each buffer's [off, off + total) bytes, of which a prompt of T rows touches only the
    // first fixed + per_row*T (a buffer carved with the defaults: all of it, at any T).  enter_prefill gives the prompt
    // exactly those ranges, so a lend bank lying wholly in a buffer's unused tail can stay filled.
    struct ArenaSpan {
        size_t off = 0, total = 0, fixed = 0, per_row = 0;
    };
    std::vector<ArenaSpan> arena_spans;
    // a window's scratch, rows of up to kWin tokens (the routed experts' rows: kWin * kPicks)
    float *h = nullptr, *h2 = nullptr, *x = nullptr, *qr = nullptr, *q = nullptr, *kvr = nullptr, *ckr = nullptr,
          *cgr = nullptr, *latent = nullptr, *ikr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr,
          *bm = nullptr, *o = nullptr, *oa = nullptr, *ao = nullptr, *mix = nullptr, *a_pre = nullptr,
          *a_post = nullptr, *a_comb = nullptr, *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr,
          *glog = nullptr, *rw = nullptr, *moe_out = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *sh_out = nullptr, *cpu_part = nullptr, *eg_x = nullptr, *eg_kv = nullptr, *logits = nullptr,
          *freqs_win = nullptr, *freqs_comp = nullptr, *xe = nullptr, *xg = nullptr, *xu = nullptr, *xhh = nullptr,
          *ed = nullptr;
    float* glog_a = nullptr;   // lookahead: the next two layers' gate logits on this layer's FFN input (rows 1024 apart)
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *tok = nullptr;
    unsigned long long* plan_ptr = nullptr;           // the decode path's VRAM experts (0: on the CPU lane)
    uint8_t* eg_rows = nullptr;
    uint16_t *emb_rows = nullptr, *ext = nullptr;   // the window's embedding rows; the window attention's rows
    AttnScratch attn_scr;   // a few queries' selected compressed rows decoded (sparse_attn)
    // the hand-off to the next device through pinned host memory (these cards copy peer to peer only through the
    // driver's staging, ~1.7 ms a copy): h | pre | topk | cand
    uint8_t* bounce = nullptr;
    cudaEvent_t ev_out = nullptr;
    float* tap = nullptr;   // a window's stream means before a DSpark target layer
    // threads / tensor parallelism: its group and its index in it, the group's all-reduces it took part in, xfer_put's
    // block counter, the hand-off to the next group (host-mapped: h | pre | topk | cand, then its flag), the window
    // MoE's grouping scratch
    int grp = 0, mi = 0;
    unsigned ar_seq = 0;
    unsigned* xcnt = nullptr;
    // the head group's members (the last layer's group): each its columns (split128 of V) of the head and the norms;
    // logits_part: a member's share of the logits (the lead's: where the others' land before they join its rows)
    x3::Mat head;
    float *norm_w = nullptr, *d_norm = nullptr, *logits_part = nullptr;
    unsigned head_seq = 0;
    // the drafter's markov bias over the head group (markov_share): this member's rows [mk_v0, mk_v0 + mk_vn) of the
    // markov head, the previous token's E row, the drafted tokens (the lead's: d_ints)
    uint16_t *mk_share = nullptr, *mk_erow = nullptr;
    int* mk_ints = nullptr;
    int mk_v0 = 0, mk_vn = 0;
    uint8_t* hand = nullptr;
    unsigned* hand_flag = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
};

// the chunked prompt path's buffers on a device: a chunk of C tokens through a layer at once (the attention in slabs of
// A queries, the indexer in slabs of AI), its GEMMs through FP16 reconstructions of the weights and cuBLAS
struct Layer;
struct Chunk {
    int C = 0, A = 0, AI = 0;
    float *h = nullptr, *x = nullptr, *acc = nullptr, *qr = nullptr, *kvr = nullptr, *ckr = nullptr, *cgr = nullptr,
          *latent = nullptr, *ikr = nullptr, *q = nullptr, *o = nullptr, *oa = nullptr, *iq = nullptr, *iw = nullptr,
          *scores = nullptr, *bm = nullptr, *mix = nullptr, *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr,
          *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr, *glog = nullptr, *rw = nullptr,
          *sg = nullptr, *su = nullptr, *y = nullptr, *eg_x = nullptr, *eg_kv = nullptr;
    uint16_t *x16 = nullptr, *qr16 = nullptr, *o16 = nullptr, *oa16 = nullptr, *lat16 = nullptr, *sh16 = nullptr,
             *h16 = nullptr, *eg_x16 = nullptr, *ext = nullptr, *w16 = nullptr, *emb = nullptr;
    size_t w16_elems = 0;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *bounds = nullptr, *row_tok = nullptr, *pos = nullptr;
    uint8_t* eg_rows = nullptr;
    int* bounds_pin = nullptr;   // pinned: the sort's bounds, read back
    // expert-parallel prefill: this device's partial output of the layers it serves as a partner (host-mapped: the
    // layer's owner sums it into its acc), the flag it is published with, and the bytes EP moved for this device
    float* ep_acc = nullptr;
    unsigned* ep_acc_flag = nullptr;
    double ep_bytes = 0;
    // the CPU lane's experts streamed in: ring_n VRAM slots of ring_stride bytes in kParts parts, a batch a part,
    // kParts batches in flight on the copy stream (filled / freed: each part's last copy, its last use)
    static constexpr int kParts = 4;
    uint8_t* ring = nullptr;
    size_t ring_stride = 0;
    int ring_n = 0;
    cudaStream_t cs = nullptr;
    cudaEvent_t filled[kParts] = {}, freed[kParts] = {};
    // in place (file_experts): each streamed expert's file region read into pinned staging (ring_n slots of
    // stage_stride bytes, a part's slots free again once its `filled` copies are done) by the readers, in parallel,
    // and DMA'd from there - a copy from the files' pages goes through the driver's staging, one thread at ~1.3 GB/s
    uint8_t* stage = nullptr;
    size_t stage_stride = 0;
    std::unique_ptr<Pool> readers;
    cublasHandle_t blas = nullptr;
    // host-mapped: a chunk's hand-off to the next group (h | pre | topk | cand), two in flight (slot = chunk % 2), each
    // with its flag, and the next group's members' acknowledgements [slot][member] (32 apart)
    uint8_t* hand[2] = {nullptr, nullptr};
    unsigned* hand_flag[2] = {nullptr, nullptr};
    unsigned* hand_ack = nullptr;
    unsigned arb_seq = 0;        // the group's chunk all-reduces
    // pinned, two chunks in flight: the embedding rows (the first layer's device), the Engram rows (an Engram layer's)
    uint16_t* emb_host[2] = {nullptr, nullptr};
    uint8_t* eg_host[2] = {nullptr, nullptr};
    cudaEvent_t emb_ev[2] = {nullptr, nullptr}, eg_ev[2] = {nullptr, nullptr};
    double stream_bytes = 0;     // the experts streamed in, in all
    // the flow (a GGUF's in-place experts, every layer of the device on it alone; DS4_FLOW=0: the batches above): a
    // (chunk, layer)'s streamed experts are those it routes to - its likely ones (the profile's chance of a pick by the
    // chunk's tokens) queued once the layer before it routed, read while that one computes, the rest of its picks once
    // it routed itself; never an expert no token of the chunk picked unless it was a likely one.  Each item read by io
    // (unbuffered: the disk straight into its staging slot, every part of every item in flight at once - none waits
    // for a batch) or taken from its RAM-tier slot as it lies, then DMA'd into the ring in order, item i into slot
    // i % ring_n; the compute takes the items as they land, in runs of what has.
    struct Item {
        int il = 0, k = 0, e = 0, v = 0, stage = -1;   // layer, chunk, expert, virtual slot, staging slot (-1: RAM)
        int mi = 0;                                    // the device's member index in the layer's group
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};   // where its gate / up / down (its parts) lie once read
        const uint8_t* ram = nullptr;                  // its RAM-tier blob, copied into staging (not pinned there)
        uint8_t* keep = nullptr;                       // (layer-major) its layer-cache slot: what is read kept there
        int64_t after = -1;                            // (layer-major) the items up to this one read first
        std::atomic<int> left{0};                      // parts still being read
        std::atomic<bool> bad{false};
    };
    bool flow = false;
    int dev = -1;                            // the device (index into devs) this chunk state belongs to
    std::unique_ptr<IoQueue> io;
    std::deque<Item> items;                  // items[0] is item `base`; the indices below are absolute
    size_t base = 0, n_read = 0, n_dma = 0, n_comp = 0;
    uint64_t n_file = 0;                     // file items so far (staging slot = ordinal % S)
    int S = 0;                               // staging slots, stage_stride apart
    std::vector<int64_t> stage_item;         // staging slot -> the item last read into it (-1: none)
    std::vector<cudaEvent_t> stage_ev, ring_fill, ring_free;   // a staging slot's DMA out; a ring slot's DMA in, its use
    int spec_k = -1, spec_il = -1, n_spec = 0;   // the likely items queued last: (chunk, layer), how many, from where
    size_t spec_begin = 0;
    std::vector<int> fl_layers, fl_T;        // the device's trunk layers in order; the run's chunk sizes
    // the row-split prefill: the replica holding each layer on this device (the flow's Layer& for il), empty otherwise;
    // rp: this prompt runs the row split - the canonical groups' source publications are off (its own wavefront's
    // buffers carry the rows, and no canonical reader takes part)
    std::vector<Layer*> lay_override;
    bool rp = false;
    int bc_n = 0, bc_w = 0;                  // a batch's compacted bounds: bc_n regions of bc_w, by batch count
    int *bc_pin = nullptr, *bc_dev = nullptr;
    uint64_t n_batch = 0;
    double wait_ms = 0;                      // the compute waiting for a read
    uint64_t it_file = 0, it_ram = 0, it_spec = 0, it_spec_unused = 0;
    uint64_t io_d0 = 0, io_b0 = 0;           // io's unbuffered / buffered bytes when the run started
    // layer-major (a run of several chunks, every layer on this device): the layers one after another, each through
    // all the chunks; a chunk's state between its layers (h | pre | topk | cand, chunk k at k chunk_hand_bytes()) in
    // host memory, and the layer cache - RAM slots where the experts a layer's chunk reads from the files are kept for
    // its later chunks, so each is read once a run
    bool lm = false, lm_pinned = false;
    uint8_t* lm_state = nullptr;
    size_t lm_state_bytes = 0;
    uint8_t* lc = nullptr;                   // lc_bytes of slots, a layer's experts' blobs side by side
    size_t lc_bytes = 0;
    int lc_il = -1, lc_used = 0, lc_n = 0;   // the layer the cache holds, its slots taken, its slots in all
    size_t lc_first = 0, lc_chk = 0;         // the layer's first item; the items before lc_chk all read
    std::vector<int> lc_of;                  // expert -> its slot (-1: none)
    std::vector<int64_t> lc_writer;          // slot -> the item reading its expert into it
    uint64_t it_lc = 0;                      // items taken from the layer cache
};

struct Layer {
    int il = 0, dev = 0, ratio = 0;
    // its group (tensor parallel: nm members, each a copy of the device state in mates, member mi's share of the
    // heads, of wo_a's groups, of wo_b's rows and of the FFNs: gate / up columns, down rows; its part of an expert
    // laid out as hlay, hblob bytes - the host state, the CPU lane and the full blob layout stay with member 0)
    int grp = 0, nm = 1, mi = 0;
    std::vector<Layer> mates;
    x3::ExpertLayout hlay;
    size_t hblob = 0;
    int ff0 = 0, ffn = 0;     // this member's FFN columns [ff0, ff0 + ffn) (split128 of n_ff; the whole one alone)
    // this member's heads [ah0, ah0 + ahn) and wo_a groups [og0, og0 + ogn) - all of them alone; in a group of more
    // members than the heads split among evenly (head_members: 9, 7 or 6 over 64 heads), the first ones none: their
    // attention adds nothing to the sum
    int ah0 = 0, ahn = 0, og0 = 0, ogn = 0;
    // the chunked prompt path: each expert's virtual slot (its VRAM slot, or n_slots + its place among the CPU lane's
    // experts, which a chunk streams in that order) - pinned on member 0, a device copy on each member
    int* vs_pin = nullptr;
    int* vs_dev = nullptr;
    std::vector<int> host_list;
    std::string name;         // "layers.N." or, for the DSpark drafter's blocks, "mtp.N."
    int n_exp = 0, k_exp = 0; // routed experts, and how many a token takes (the drafter's: 128, 3)
    bool mtp = false;         // a DSpark block (index stage = il - n_layers)
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1;      // the kv source whose compressed KV this layer reads (index into Impl::srcs)
    int ik_src = -1;   // the kv source whose index keys its indexer reads
    int eg = -1;       // engram layer index, or -1
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_bias = nullptr, *eg_w = nullptr,
          *gate_bias_vl = nullptr;   // (the image tokens' routing bias, with the vision tower)
    uint16_t* i_wproj = nullptr;
    float* gate_w = nullptr;   // the router's weights, F32 (DeepSeek's inference computes the gate in F32)
    uint8_t* win = nullptr;                     // [win] rows of kKvFp8 (kv_row_bytes)
    float *c_kv = nullptr, *c_score = nullptr;  // the compressor's partial group (ratio > 1)
    // a run of chunks' mid-run checkpoint (feed_mark): the ring and the partial group as they were before a row
    uint8_t* snap_win = nullptr;
    float *snap_kv = nullptr, *snap_sc = nullptr;
    // routed experts
    x3::ExpertLayout lay;
    size_t blob = 0;                            // bytes of one expert's blob
    int n_vram = 0;
    uint8_t* vram_base = nullptr;
    // lending: slots [0, n_perm) live in vram_base (its own cudaMalloc, the drive's spare included); slots
    // [n_perm, n_perm + n_lend) live in the device's prompt arena (lend_base) and exist only between prompts - a
    // prompt evicts them (n_slots falls back to n_perm) and the refill after it fills them again.  A drafter block
    // (mtp) keeps all of its slots there: n_perm = 0, n_lend = its count.
    int n_perm = 0, n_lend = 0;
    uint8_t* lend_base = nullptr;
    bool lend = false;                          // its banks are in the arena: emptied before a prompt, filled after it
    // the lend slots a prompt kept (their bytes lay wholly in the chunk buffers' unused tails): (expert, slot), hidden
    // from the prompt's slot map and put back after it - exit_prefill refills only the ones the prompt took
    std::vector<std::pair<int, int>> lend_keep;
    std::vector<int> slot;                      // expert -> VRAM slot or -1
    std::vector<int> ram;                       // expert -> RAM-tier slot or -1 (a GGUF's in-place experts; empty: none)
    std::vector<uint8_t*> host;                 // expert -> host (non-null: on the CPU lane; its halves in hh)
    // the CPU lane keeps an expert as NH column parts, part X on NUMA node X: gate / up columns [X n_ff / NH, ..),
    // down columns [X dim / NH, ..), each a standalone EXL3 matrix (its own trellis, the full suh, its part of svh)
    struct Half {
        size_t trellis[3], suh[3], svh[3], bytes;
        int k[3], n[3];
    } hl;
    std::vector<std::array<uint8_t*, 4>> hh;     // expert -> its parts on the nodes
    // the expert cache: one VRAM slot and one set of host parts kept free, so an expert can move in while the one it
    // replaces moves out; heat is how much each expert was used lately (the profile's share to start with)
    int n_slots = 0, spare_vram = -1;
    std::array<uint8_t*, 4> spare_host = {nullptr, nullptr, nullptr, nullptr};
    std::vector<float> heat;
    std::vector<int> cpu_hits;                  // CPU-lane experts used since the cache last looked
    bool swap_pending = false;
    unsigned long long* tab_pin = nullptr;      // pinned: the table entries a swap writes
    // a speculative window's backups (trunk layers): the ring rows it replaced, and a compressor's state before it and
    // the window's raw rows (kv, then score) to replay the accepted ones
    uint8_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
    unsigned long long* vram_tab = nullptr;     // device [n_exp]
    std::vector<std::array<x3::Mat, 3>> vm;     // expert -> its gate, up, down as EXL3 views of its VRAM blob
    std::vector<uint64_t> region_off;           // expert -> file region (offset, bytes, file)
    std::vector<uint64_t> region_bytes;
    std::vector<int> region_file;
    size_t piece_src[3][3];                     // part (gate, up, down) x (trellis, suh, svh): offset in the region
    size_t piece_bytes[3][3];
    size_t dense_bytes = 0;
    // a GGUF's MXFP4 experts (lay.fmt == gg::kFmtMXFP4): matrix r (gate, up, down) of expert e lies at mx_off[r] +
    // e mx_stride[r] in file mx_file[r] - loaded from there, and read in place (the file's mapping) by the CPU lane
    int mx_file[3] = {-1, -1, -1};
    uint64_t mx_off[3] = {0, 0, 0}, mx_stride[3] = {0, 0, 0};
};

struct Src {
    int layer = 0, ratio = 1;
    int64_t cap = 0;
    // per device (nullptr where no layer reads it): rows of kKvFp4 (the compressed KV), kKvFp4p (the index keys)
    std::vector<uint8_t*> ckv, ik;
    // the row-split prefill's copies on every device (DS4_ROW_SPLIT only, allocated once at load; they stand in for
    // ckv/ik while a row-split prompt runs - a prompt's wavefront fills each device's own rows)
    std::vector<uint8_t*> ckv_rp, ik_rp;
    uint8_t *h_ckv = nullptr, *h_ik = nullptr;   // pinned: a window's new rows, on their way to the other devices
    cudaEvent_t ev_ckv = nullptr, ev_ik = nullptr;
    // threads: whether devices outside the source's group read it, and the host-mapped hand-off of a window's new rows
    // (ckv rows, then ik rows at kWin * hd) and its flag
    bool remote = false;
    uint8_t* x_stage = nullptr;
    unsigned* x_flag = nullptr;
    // pipelined chunks: a second staging slot (slot 1; slot 0 is x_stage), its flag, and the readers' acknowledgements
    // [slot][device] (32 apart); readers: the devices outside the source's group that read it
    uint8_t* x_stage1 = nullptr;
    unsigned* x_flag1 = nullptr;
    unsigned* x_ack = nullptr;
    std::vector<int> readers;
    uint8_t* stage(int slot) const { return slot ? x_stage1 : x_stage; }
    unsigned* flag(int slot) const { return slot ? x_flag1 : x_flag; }
};

// a host thread that enqueues one device's work (threads mode)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu);
        return !busy;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

struct Group {
    std::vector<int> mem;   // device indices, member order (member 0 the lead)
    ArGroup ar, arb;        // a window's all-reduces, a chunk's
    // the head group: the members' logits shares on their way to the lead, two slots (by seq), their flags and the
    // lead's acknowledgements (host-mapped)
    float* lg_stage = nullptr;
    unsigned *lg_flag = nullptr, *lg_ack = nullptr;
};

constexpr int kMaxL = 64;   // layers (the trunk's and the drafter's) a window plan covers

// a row-split chunk's publications for one layer: its ring and compressed rows, and (at a source's writer) the rows it
// completed.  They are final at the layer's compressor, so the layer emits them there instead of after the whole layer -
// the chunks behind wait for these bytes, not for this chunk's attention and experts
struct RpPub {
    Xfer ring, comp, src;
    unsigned* ring_flag = nullptr, *comp_flag = nullptr, *src_flag = nullptr;
    unsigned ring_seq = 0, comp_seq = 0, src_seq = 0;
};

// what a window's layers need from the host, decided before they run (threads enqueue the layers concurrently): the
// mailbox seq of each layer (async: the device publishes its routing), the indexer's selection sizes and the candidate
// blocks as each layer sees them, the window's hand-off seq
struct WinPlan {
    bool par = false, async = false;
    unsigned win = 0;
    int k = 0;              // (a run of chunks) the chunk's index: its hand-off slot k % 2
    int snap_t = -1;        // (a run of chunks) the chunk's row whose state before it the layers keep (-1: none)
    int eg_slot = -1;       // (a run of chunks, layer-major) the Engram rows' slot (-1: k % 2)
    unsigned vl = 0;        // (a window) bit t: token t is inside a picture's span (routed with the image bias)
    const RpPub* rp_pub = nullptr;   // (the row split) the layer publishes these at its compressor (nullptr: after it)
    unsigned seq[kMaxL] = {};
    int topk[kMaxL][kWin] = {}, cand[kMaxL][kWin] = {};
};

}  // namespace

// ------------------------------------------------------------------ the model
struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    std::vector<Dev> devs;
    // the VRAM each device leaves free after the load (resolved from --reserve-mib / --reserve-pct / the default)
    std::vector<size_t> resv;
    std::vector<Layer> L;
    std::vector<Src> srcs;
    std::vector<Group> groups;
    std::vector<std::unique_ptr<Worker>> workers;   // threads mode: one per device
    bool par = false;                 // threads mode (the windows: decode)
    bool prompt_par = false;          // the prompt path's chunks (a thread a GPU), as --tp / --threads asked
    unsigned win_seq = 0;             // the windows the threads ran (the hand-offs' seq)
    // the chunked prompt path (threads mode): chunk tokens a pass, runs shorter than chunk_min in windows
    int chunk = 0, chunk_min = 512;
    // each group's step for the prompt chunk (one entry per group, non-decreasing, each a whole multiple of the one
    // before it): all `chunk` without --prefill-chunk-max; decode never reads it
    std::vector<int> chunk_steps;
    static constexpr int kFlowChunkMin = 2;   // chunk_min with every device's layers on the flow (see load)
    // --prefill-chunk at most (route() runs in kVlRows-row slabs).  The default stays kVlRows: a 16384-token chunk
    // reads a 16K prompt's experts once instead of twice (V100: 461.8 -> 321.3 s, 736 -> 506 GB read) but its buffers
    // take ~1.9 GB of VRAM experts (745 -> 644 on a 32 GB V100) from every decoded token
    static constexpr int kMaxChunk = 4 * kVlRows;
    // the split search's planning margin per device (MiB): the 600 MiB default reserve, kept here so a percentage
    // reserve (--reserve-pct) changes the experts' placement, not which layers go to which card
    static constexpr int kSplitReserveMib = 600;
    std::vector<std::unique_ptr<Chunk>> chunks;   // per device
    // ---- the row-split prefill (--row-split-prefill / DS4_ROW_SPLIT): every device holds the same K-layer group and
    // runs its own chunk of the prompt's rows through it, stage by stage - the groups swapped between stages, each
    // stage's layer state published to the chunks behind it as it lands (a wavefront over (chunk, layer))
    struct RpRep {
        Layer l;      // the replica: its dense weights in arena, its ring / compressor state allocated once at load
        Bump bump;    // the arena the stage's dense weights are carved from
        // (the slot's own compressor state, kept here: the replica Layer is overwritten at every stage, and a stage
        // layer without a compressor would drop the pointer it holds while the allocation stays)
        float* c_kv = nullptr, *c_score = nullptr;
    };
    struct RpRun {   // a prompt's partition: one chunk a device, the head's last
        std::vector<int> order;       // chunk index -> device index
        std::vector<int> T;           // the chunks' rows
        std::vector<int64_t> p0;      // their first positions
        std::vector<int> chunk_of;    // device -> its chunk (-1: none)
        int64_t pos0 = 0;
        int n = 0, rows = 0;          // the prompt's first position, its rows, its widest chunk
        // a prompt over the devices' caps runs in rounds of them (the last one the remainder): this round, of how
        // many, and whether it is the last - only the last round's last chunk computes logits and takes the taps
        int round = 0, nround = 1;
        bool final = true;
    };
    struct RpPlan {
        bool on = false;
        int k = 0, ng = 0;                                   // layers a group, groups
        std::vector<size_t> slot_bytes;                      // a slot's arena, in all
        std::vector<std::vector<RpRep>> rep;                 // [device][slot]
        std::vector<std::vector<uint8_t*>> ring_pub;         // [device][layer]: its rows the chunk published
        std::vector<std::vector<unsigned*>> ring_flag;
        std::vector<std::vector<uint8_t*>> comp_pub;         // [device][layer]: the compressor's partial group
        std::vector<std::vector<unsigned*>> comp_flag;
        std::vector<std::vector<uint8_t*>> src_pub;          // [device][source]: its ckv rows, then its ik rows
        std::vector<std::vector<unsigned*>> src_flag;
        std::vector<std::vector<int>> src_rows;              // [device][source] the publish buffer's rows
        std::vector<char> src_ik;                            // [source] its index keys exist at all
        std::vector<std::vector<uint8_t*>> ckv_can, ik_can;  // [source][device] the canonical marks
        std::vector<int*> vs_pin;                            // [device] every expert a streamed one
        std::vector<int*> vs_dev;
        std::vector<std::vector<char>> pulled;               // [device][source] this prompt's rows there
        std::unique_ptr<std::atomic<int>[]> prog;            // [device] the last layer it finished (the LIVE line)
        bool src_swapped = false;                            // this prompt's sources are its own copies (see rp_end)
        unsigned seq = 0;                                    // the prompt's flag base
        // the balance: the prompt's rows go to the cards in proportion to their weights, not equally - a card that
        // runs a layer faster takes more rows, so the cards finish each layer together and the wavefront waits less.
        // The weights are learned from the cards' own layer time (events around each layer, the wait for the chunk in
        // front excluded) and hold at 1 (the even split) until a round has measured them
        std::vector<std::vector<cudaEvent_t>> ev;            // [device][2 * slot]: a layer's start, its end
        std::vector<double> work_ms;                         // [device] this round's summed layer time
        std::vector<double> row_w;                           // [device] its share's weight (mean 1)
        bool balance = true;                                 // DS4_ROW_SPLIT_BALANCE=0 holds the weights still
        bool early_pub = true;                               // DS4_ROW_SPLIT_EARLY_PUB=0 publishes after the layer
    } rp;
    RpRun rp_run;
    unsigned rp_prompt_seq = 0;
    // expert-parallel prefill (--ep-prefill): the share of each layer's non-resident experts every device streams,
    // its pinned virtual-slot map, the run's x | ridx | rw publication (host-mapped, one prompt at a time) and the
    // flag it is announced with; each partner's partial acc comes back through its chunk's ep_acc
    struct EpShare {
        int di = -1, n = 0, n_slots = 0;   // the device; its experts; the VRAM slots it runs first (the owner's)
        std::vector<int> list;             // the share's experts, ascending: its virtual slots in this order
        int* vs_pin = nullptr;             // [n_exp] pinned: expert -> n_slots + its place in list, or -1
        int* vs_dev = nullptr;             // [n_exp] device
    };
    std::vector<std::vector<EpShare>> ep_sh;   // [layer][device]
    // one publication a device (the layers it owns write it): an owner's next publication follows its own ep_collect of
    // the layer before, so a partner reading layer il cannot see a later owner's x overwrite it
    std::vector<float*> ep_x;          // [device] host-mapped: Cmax x dim (the owner's normalized layer input)
    std::vector<int*> ep_ridx;         // Cmax x kPicks (its routing)
    std::vector<float*> ep_rw;         // Cmax x kPicks (the picks' weights)
    std::vector<unsigned*> ep_x_flag;  // [device] the publication's flag
    bool ep_alloc = false;                     // the staging and the shares' maps exist
    bool ep_run = false;                       // this prompt runs expert-parallel
    unsigned ep_base = 0, ep_n = 0;            // the run's flag sequence base (a prompt apart from every other)
    // DS4_PASS_LEND: a pair of layers whose banks are one block (place_experts: an encoder layer's and a decoder-tail
    // layer's, on one device, the tail's above the encoder's), and the layers a prompt's passes have borrowed from or
    // for, with the decode addressing pl_restore puts back
    struct PlPair {
        int enc = -1, tail = -1;
        uint8_t* base = nullptr;               // the block: enc's bank at base, the tail's at base + tail_off
        size_t tail_off = 0;
        int n_enc = 0, n_tail = 0;             // (the block's slots for each side, its own expert bytes apart)
        int ne = 0, nt = 0;                    // (the two banks the decoder keeps)
    };
    std::vector<PlPair> pl_pairs;
    struct PlSaved {
        int il = -1;
        uint8_t* vram_base = nullptr;
        int n_perm = 0;
    };
    std::vector<PlSaved> pl_saved;             // (the borrows in force: emptied when they are put back)
    size_t eg_cap = kWin;             // tokens h_eg holds
    int tap_cap = kWin;               // rows h_taps holds (a chunk seeds the drafter with its last 128)
    bool logits_all = false;          // the last feed was a chunk with logits for every row (chunk_logits)
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;   // a run of chunks: the tokens done so far (the main thread, while it waits)
    std::vector<cudaEvent_t> chunk_done;     // on the last layer's device: chunk k through every layer
    std::unique_ptr<std::atomic<int>[]> chunk_rec;   // chunk k's event recorded
    // a prompt the flow could take: the flow or windows of 16, whichever this PC runs it faster (pick_path) - this
    // PC's speeds measured at load (calibrate_paths), each path's predictions scaled by how long it really took
    struct PathModel {
        bool on = false;
        double dma = 0, copy = 0, disk = 0, cpu = 0, cpu_cold = 0;   // bytes/s (disk, cpu_cold 0: not measured)
        // the flow's taken / predicted so far [0] by the prompt's size class (2-31, 32-127, 128-511, 512 tokens and
        // more: a short one's fixed costs weigh more), and how many prompts set it ([1] unused)
        double k[2][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}};
        int64_t kn[2][4] = {};
        double win_s = 0;                                              // windows: seconds a window took, so far
        int64_t win_n = 0;
        int last_cls = 0, last_nw = 0;                                 // the last pick's size class, its windows
        double m = 1;                                                  // the picks' spread, against the profile's
        int64_t n_m = 0;
        int last = -1;                                                 // the last pick: 1 the flow, 0 windows
        double last_pred = 0;                                          // its prediction (s, before its k)
    } pm;
    static constexpr int kPickMax = 2048;   // (from here on the flow: windows' work grows with every 16 tokens)
    static constexpr double kPickRes = 0.9;  // (this share of the CPU-lane experts in RAM: windows if predicted faster)
    std::atomic<int> lm_layers{0};           // (layer-major) the layers every chunk went through
    WinPlan wplan;
    int head_dev = 0;
    x3::Mat head;
    float* norm_w = nullptr;          // on the head device
    // the drafter: main_proj / main_norm (on its first block's device), its final norm, the markov head and the
    // confidence head (on the head device: the drafts' logits come from the trunk's head)
    x3::Mat d_main_proj;
    float *d_main_norm = nullptr, *d_norm = nullptr, *d_conf = nullptr;
    uint16_t *d_mk_embed = nullptr, *d_mk_head = nullptr;   // BF16 [vocab][rank], F16 [vocab][rank]
    // a head group of two or more: the markov steps over its members (each its rows of the bias and an argmax of its
    // share, the global one through host-mapped slots [step][member] {value, index, seq}) in place of the lead's whole
    // rows - E host-mapped (each member reads the picked token's row), a draft's seq in mk_seq
    bool mk_split = false;
    uint16_t* h_mk_embed = nullptr;
    unsigned* mk_slots = nullptr;
    unsigned mk_seq = 0;
    int d_dev = 0;                    // the drafter's first block's device
    bool spec_win = false;            // the window being run is a draft's verification (layer() keeps backups)
    float* h_taps = nullptr;          // pinned [kWin][targets][dim]: the last window's taps
    cudaEvent_t ev_seed = nullptr;    // the drafter's seeding read h_taps: the next window's taps wait for it
    float* d_confv = nullptr;         // on the head device
    int* d_ints = nullptr;            // on the head device: drafts in, picks out
    float conf_threshold = std::getenv("DS4_DSPARK_CONF") ? (float) std::atof(std::getenv("DS4_DSPARK_CONF")) : 0.5f;
    // a sampled round's drafts: the distribution q each one was drawn from (its kept list, on the head device) for
    // verify()'s rule - a point at the draft for the drafter's argmax; and the chain they were drawn with (verify()
    // checks it is its own)
    strata::kernels::SpecList* d_specq = nullptr;
    std::vector<strata::kernels::SpecList> h_specq = std::vector<strata::kernels::SpecList>(kWin);
    bool drafts_q = false;
    float dq_temp = 0.0f, dq_top_p = 1.0f, dq_min_p = 0.0f;
    int dq_top_k = 0;
    uint64_t dq_seed = 0, dq_counter = 0;
    // (tests) DS4_DSPARK_SAMPLE=1: a sampled request's drafts drawn from the drafter's distribution through the
    // request's chain, as DeepSeek's reference does (else its argmax: the same acceptance on Uranus, 31-32% of 5
    // drafts a round either way); DS4_DSPARK_QTEMP=f: those draws at f x the request's temperature (any q keeps the
    // rule exact); DS4_DSPARK_MAX=n: at most n drafts a round; DS4_DSPARK_DUMP=file: the drafter's inputs and outputs
    // (tools/ds4/dspark_ref.py checks them against DeepSeek's checkpoint)
    bool sample_drafts = std::getenv("DS4_DSPARK_SAMPLE") != nullptr && std::atoi(std::getenv("DS4_DSPARK_SAMPLE")) != 0;
    float q_temp_scale = std::getenv("DS4_DSPARK_QTEMP") ? (float) std::atof(std::getenv("DS4_DSPARK_QTEMP")) : 1.0f;
    int max_drafts = std::getenv("DS4_DSPARK_MAX") ? std::atoi(std::getenv("DS4_DSPARK_MAX")) : 0;
    std::FILE* dump_d = std::getenv("DS4_DSPARK_DUMP") ? std::fopen(std::getenv("DS4_DSPARK_DUMP"), "wb") : nullptr;
    float* h_dump = nullptr;          // (the dump's pinned copies: each block's stream, each block's ring)
    uint16_t* h_dump_ring = nullptr;

    uint16_t* embed_pinned = nullptr;
    std::vector<float> freqs_win, freqs_comp;
    std::vector<std::unique_ptr<Pool>> pools;   // one per NUMA node, its workers on that node's cores
    std::vector<std::vector<int>> cores;        // each node's physical cores
    std::vector<std::vector<int>> all_cpus;     // each node's logical CPUs (the siblings too)
    int NH = 1;                                 // the CPU lane's parts of an expert (the NUMA nodes)
    // the CPU lane's experts read in place from the model's files (one part, the files' layout; the page cache keeps
    // what fits) instead of copied into RAM, and how much of them the load reads ahead (the most-routed first)
    bool file_experts = false;
    uint64_t warm_bytes = 0;
    static constexpr uint64_t kRamMargin = 8ull << 30;   // RAM left to the rest (the system, the page cache's churn)
    uint8_t* file_part(const Layer& l, int e) const {
        return pk.maps[(size_t) l.region_file[(size_t) e]].first + l.region_off[(size_t) e];
    }
    // host side of the CPU lane, a window's rows
    float *h_x = nullptr, *h_w = nullptr, *h_part = nullptr;
    int* h_idx = nullptr;
    uint8_t* h_eg = nullptr;          // the window's Engram rows: [engram layer][token] of [cols][hd] then the scales
    // the Engram rows' reader (its own threads and file handles), the rows already fetched, the window's batches (one
    // an Engram layer: asked for when the window's tokens are known, collected right before that layer)
    EgReader eg_io;
    EgRowCache eg_cache;
    std::mutex eg_cache_mu;
    std::vector<std::unique_ptr<EgBatch>> eg_win;
    std::vector<uint8_t> eg_q6k;   // Engram layer li's table in ggml Q6_K (a row one superblock a 256 values), else FP8
    std::atomic<int64_t> eg_rows_n{0}, eg_hits_n{0}, eg_reads_n{0}, eg_wait_ns{0}, eg_io_ns{0};
    uint16_t* h_emb = nullptr;        // the window's embedding rows
    std::vector<float> c_xh, c_xsb, c_acc, c_xhd, c_xsd, c_accd, c_yd;
    // the state
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    struct Check {                    // the state at a position: every window ring, the compressors' partial groups
        int64_t pos = -1;
        std::vector<std::vector<uint8_t>> win;
        std::vector<std::vector<float>> comp;
    };
    Check ck[2];                      // the end of the last prompt, and the position before it
    std::vector<uint32_t> prof;       // [nl][n_exp] routed-expert hits
    Stats st;
    int cur_topk_n[kWin] = {}, cur_cand_n[kWin] = {};   // what the latest indexer / candidate source left, per query
                                                        // (plan_window's: the windows' plans are made in order)
    int n_logits = 0;                 // the last forward's logit rows
    // the decode path (windows of one): the device publishes each layer's routing into reqs, the service thread
    // answers in resps with the CPU lane's part, and the device waits for the answer only before the expert sum
    Req* reqs = nullptr;
    Resp* resps = nullptr;
    unsigned seq_issued = 0;
    std::atomic<int> layer_of[kRing];
    std::thread service;
    std::atomic<bool> quit{false};
    bool sync_decode = std::getenv("DS4_SYNC") != nullptr;
    // a window of one through moe_gate_up / moe_down (DS4_MOE_TOKEN=0: win_experts, whose down rows run
    // before the wait for the CPU lane - the same bits, no faster)
    bool moe_token_path = std::getenv("DS4_MOE_TOKEN") == nullptr || std::atoi(std::getenv("DS4_MOE_TOKEN")) != 0;
    int async_max_t = std::getenv("DS4_ASYNC_T") ? std::atoi(std::getenv("DS4_ASYNC_T")) : 8;   // windows up to this
    // lookahead: while layer L runs, the device guesses the next layer's picks (its gate on L's FFN input, the same
    // choice) and publishes them with L's routing; the CPU lane asks for the guessed experts it will read from the files
    // at once (a thread of its own, os::prefetch), so the SSD reads them while L computes instead of on the next
    // layer's critical path.  Only the page cache sees it - the outputs are the same.  DS4_AHEAD=0: off, 2: also the
    // layer after (from L's input too); DS4_AHEAD_EXTRA: guesses past the next layer's top k (< 0: fewer - the guess's
    // picks by rank on Sun: 97 91 81 70 57 41 %, past the top 6 29 20 %: reads wasted on an SSD-bound PC).
    int ahead_mode = std::getenv("DS4_AHEAD") ? std::atoi(std::getenv("DS4_AHEAD")) : 1;
    int ahead_extra = std::getenv("DS4_AHEAD_EXTRA") ? std::atoi(std::getenv("DS4_AHEAD_EXTRA")) : 0;
    // (DS4_AHEAD_ORDER=0: the CPU lane computes a layer's experts in expert order, not the ones asked for early first)
    bool ahead_order = std::getenv("DS4_AHEAD_ORDER") == nullptr || std::atoi(std::getenv("DS4_AHEAD_ORDER")) != 0;
    std::vector<uint8_t> ahead_fl;     // [layer][1024]: 1 the top-k guess, 2 any guess, a layer before; 4 two layers before;
                                       // 8 asked for (prefetch)
    std::vector<uint8_t> ahead_rank;   // [layer][1024]: 1 + the expert's rank in the guess a layer before (0: not guessed)
    std::vector<unsigned> ahead_for;   // [layer]: the request (seq) its flags are for
    std::thread pf_th;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    std::vector<std::pair<const void*, size_t>> pf_q;
    // the expert cache: swaps prepared on threads of their own (DS4_CACHE_THREADS, 1 by default - 4 measured no faster
    // on one V100 - more for a PC to try: each its staging buffers and
    // copy streams), switched between windows
    struct Swap {
        int layer = 0, in = -1, out = -1;
        int ram = -1;   // (copy_mode, `in` held in RAM) the spare slot `out` is copied into from VRAM
    };
    static constexpr int kMaxSwaps = 8;
    // the swap rule: a CPU-lane expert in when its heat passes the coldest VRAM one's x margin + bonus; heat x decay
    // every 256 tokens; at most `inflight` swaps under way (DS4_CACHE_MARGIN / _BONUS / _DECAY / _INFLIGHT)
    static float env_f(const char* k, float d) { return std::getenv(k) ? (float) std::atof(std::getenv(k)) : d; }
    const float c_margin = env_f("DS4_CACHE_MARGIN", 1.25f), c_bonus = env_f("DS4_CACHE_BONUS", 1.0f),
                c_decay = env_f("DS4_CACHE_DECAY", 0.9f);
    const int c_inflight = std::max(1, (int) env_f("DS4_CACHE_INFLIGHT", (float) kMaxSwaps));
    // the RAM tier (a GGUF's in-place experts, as GLM's: the hottest in VRAM, the next hottest here, the coldest in the
    // files): RAM slots of one expert blob each, filled at load with the hottest experts VRAM does not hold, so the
    // CPU lane reads them from RAM; between windows (cache_tick) a hotter expert from the files takes the place of the
    // layer's coldest held one - read into a free slot by ram_th, switched while nothing reads them.  Opt-in
    // (STRATA_DS4_RAM_TIER=1): on a V100 with 64 GB of RAM it was slower than the page cache alone (1.4 against 2.2-2.4
    // tokens/s sampled) - the cache follows a conversation's experts at once, the tier's moves lag behind.
    struct RamSwap {
        int layer = 0, in = -1, out = -1, slot = -1;
    };
    std::vector<uint8_t*> ram_slots, ram_chunks;
    std::vector<size_t> ram_chunk_bytes;
    bool ram_pinned = false;   // the slots are pinned memory (the prompt path DMAs them from where they lie)
    size_t flow_part[3] = {0, 0, 0};   // the flow's staging room for each part: its largest over the trunk's layers
    std::vector<int> ram_free;
    std::deque<RamSwap> ram_queue;
    std::vector<RamSwap> ram_ready;
    std::condition_variable ram_cv;
    std::thread ram_th;
    int ram_inflight = 0, ram_held = 0;
    bool ram_tier = std::getenv("STRATA_DS4_RAM_TIER") != nullptr && std::atoi(std::getenv("STRATA_DS4_RAM_TIER")) != 0;
    // STRATA_DS4_DISK_SKIP_W (experimental opt-in, CHANGES THE ANSWERS; 0 = off): on the small windows (decode, a
    // draft's check - never the prompt path) the CPU lane leaves out a routed expert whose share of its token's route
    // weights is below this and whose pages are not in RAM now (the page cache or the RAM tier; os::in_ram - Linux every
    // page, Windows sampled pages of the working set), so a token waits on the SSD only for the experts that weigh; the
    // weight it carried is lost (no renormalisation).  On layers from STRATA_DS4_DISK_SKIP_FROM (0) on.  An expert left
    // out still counts as used (the expert cache, the profile).
    const float disk_skip_w = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_W");
        return v != nullptr ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    const int disk_skip_from = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_FROM");
        return v != nullptr ? std::max(0, std::atoi(v)) : 0;
    }();
    bool in_ram(const Layer& l, int e) const {   // (DISK_SKIP_W) expert e's pages in RAM now
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) return true;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r)
                if (!os::in_ram(mx_mat(l, e, r), mx_part_bytes(l, r))) return false;
            return true;
        }
        return !file_experts || os::in_ram(file_part(l, e), l.blob);
    }
    // a GGUF's CPU-lane experts copied into RAM (copy_experts, after VRAM took its share: --mmap-experts off - auto reads
    // them in place; DS4_COPY_GB=<n>: the hottest n GB, the rest in place - static, no moves between
    // them): RAM slots (l.ram, as the tier's) of NH parts, part X on NUMA node X with rows [X n / NH, (X + 1) n / NH) of
    // each matrix at cp_lay[layer][r] - whole rows, so the split never cuts a GGUF block (MXFP4's 32, the IQ / K types'
    // 256 values: blocks run along a row); one spare slot a layer for the cache's swaps (the expert VRAM gives back takes
    // the RAM place of the one it takes in)
    // Windows' commit on an integrated GPU: its allocations are charged to the commit (RAM + page file) like any private
    // memory - measured on a Radeon 8065S with 160 GiB of its own: the system's free commit fell by each one, whatever
    // its size (one above ~3 GiB also counts as the process's private bytes, a smaller one does not, the commit the
    // same), so a page file at least the GPU's share is what lets it hold experts at all.  The experts it holds
    // (place_experts) and the copies stay within what Windows can still commit, less max(8 GB, 10% of the RAM) for the
    // other programs - commit_kept: the experts that kept off the GPU, pagefile_hint: the page file (GB) that would hold
    // them (the log's WARNING, a NOTE and INFO for the server)
    double commit_kept = 0, pagefile_hint = 0;
    double commit_margin() const {
        const os::Commit c = os::commit();
        return std::max(8.0 * (double) (1ull << 30), c.ram == ~uint64_t{0} ? 0.0 : 0.1 * (double) c.ram);
    }
    // what may still be committed beside the margin (~inf: nothing limits it; DS4_COMMIT_CAP_GB: at most that, tests)
    double commit_room() const {
        const os::Commit c = os::commit();
        double avail = c.avail == ~uint64_t{0} ? 1e300 : (double) c.avail;
        if (const char* v = std::getenv("DS4_COMMIT_CAP_GB")) avail = std::min(avail, std::atof(v) * 1e9);
        return avail >= 1e300 ? 1e300 : std::max(0.0, avail - commit_margin());
    }
    // a device whose allocations Windows charges to the commit: an integrated GPU (DS4_COMMIT_CAP_GB: any, tests)
    bool commit_charged(const Dev& d) const {
        if (std::getenv("DS4_COMMIT_CAP_GB") != nullptr) return true;
#if defined(_WIN32)
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, d.id) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        return v != 0;
#else
        (void) d;
        return false;
#endif
    }
    bool copy_mode = false;
    std::vector<std::array<uint8_t*, 2>> ram_part;   // slot -> its parts
    std::vector<std::array<size_t, 4>> cp_lay;       // layer -> part offsets of gate, up, down rows; a part's bytes
    std::vector<int> copy_spare;                      // layer -> its spare slot (-1: none)
    std::mutex sw_mu;
    std::condition_variable sw_cv;
    std::deque<Swap> sw_queue;
    std::vector<Swap> sw_ready;
    struct SwapBufs {
        uint8_t *a = nullptr, *b = nullptr, *h = nullptr;
        std::vector<cudaStream_t> st;   // a copy stream a device
    };
    std::vector<SwapBufs> sw_bufs;
    std::vector<std::thread> cache_ths;
    int sw_inflight = 0;
    int64_t decay_tokens = 0;
    // the arena's two layouts: 0 the prompt's (the chunk buffers and ring are its bytes; the trunk's experts are in
    // their own banks, the drafter's offline), 1 the decoder's (the trunk's in their banks and the arena's lend slots,
    // the drafter's in the arena).  enter_prefill() before a run of chunks, exit_prefill() after it (and before the
    // first window of a decode); draining: the cache applies what its threads finished and starts nothing new
    std::atomic<int> phase{0};
    std::atomic<bool> draining{false};
    double lend_ms = 0;      // (exit_prefill's refill: its experts, bytes, time - the log line)
    int64_t lend_n = 0;
    double lend_gb = 0;
    std::vector<cudaStream_t> copy_st;   // the lend refill's copies (exit_prefill), one a device
    bool check_swaps = std::getenv("DS4_CHECK_SWAPS") != nullptr;
    std::vector<float> host_logits;
    std::vector<void*> host_allocs;
    std::FILE* dump_h = std::getenv("DS4_DUMP_H") ? std::fopen(std::getenv("DS4_DUMP_H"), "wb") : nullptr;
    // DS4_DUMP_L: (threads) every layer's residual streams as records [layer, p0, T, T x hc x dim floats], window or
    // chunk path (tests: where the two part)
    std::FILE* dump_l = std::getenv("DS4_DUMP_L") ? std::fopen(std::getenv("DS4_DUMP_L"), "wb") : nullptr;
    // DS4_DUMP_R: the chunk path's routing, a record a layer [layer, p0, T, k, n_exp, n_vs, n_slots, kPicks] (int64),
    // then the picks' experts [T kPicks] (the first k of each token's), each expert's virtual slot [n_exp], the
    // slots' row bounds [n_vs + 1], the sorted rows' tokens [T k] (int32), the gate's logits [T n_exp] and its bias
    // [n_exp] (F32)
    std::FILE* dump_r = std::getenv("DS4_DUMP_R") ? std::fopen(std::getenv("DS4_DUMP_R"), "wb") : nullptr;
    // DS4_DUMP_Y=<layer>:<file> (layer -1: every layer): its expert sets on the chunk path, a record a piece [layer,
    // row0, rows, n_exp, wbase kind (0 VRAM, 1 ring), max_rows] (int64), its bounds [n_exp + 1], its h rows [rows ff]
    // (F16) and y rows [rows dim] (F32)
    int dump_y_layer = std::getenv("DS4_DUMP_Y") ? std::atoi(std::getenv("DS4_DUMP_Y")) : -1;
    std::FILE* dump_y = [] {
        const char* v = std::getenv("DS4_DUMP_Y");
        const char* f = v != nullptr ? std::strchr(v, ':') : nullptr;
        return f != nullptr ? std::fopen(f + 1, "wb") : nullptr;
    }();
    std::mutex dump_mu;
    // DS4_PROF: the GPU's time a decoded token by stage - events in stream order at a layer's start, after its
    // attention, after the routing and the shared expert, after the experts' gate / up, after the wait for the CPU
    // lane, after the down pass and the post - read after the token's sync, each the time since the mark before it
    bool prof_on = std::getenv("DS4_PROF") != nullptr;
    struct ProfMark {
        const Dev* d;
        cudaEvent_t ev;
        int stage;
    };
    std::mutex prof_mu;
    std::vector<ProfMark> prof_marks;
    // (an event belongs to the device that made it: recorded on another device's stream it is an invalid resource
    // handle - a stale error the next kernel's check reports - so the pool is a device's own list, never shared)
    std::vector<std::vector<cudaEvent_t>> prof_pool = std::vector<std::vector<cudaEvent_t>>(32);
    bool prof_window = false;   // (decode windows only: T == 1)
    // ... and a run of chunks' (prof_chunk): a layer's start, after its attention, after the routing and the shared
    // expert, after its experts (the streamed ones' waits for their reads in it); printed after the run
    bool prof_chunk = false;
    double cprof_ms[4] = {};
    void prof_mark(const Dev& d, int stage) {
        if (!prof_window && !prof_chunk) return;
        std::lock_guard<std::mutex> lk(prof_mu);
        std::vector<cudaEvent_t>& pool = prof_pool[(size_t) d.id % prof_pool.size()];
        cudaEvent_t e = nullptr;
        if (pool.empty()) cudaEventCreate(&e);
        else {
            e = pool.back();
            pool.pop_back();
        }
        cudaEventRecord(e, d.st);
        prof_marks.push_back({&d, e, stage});
    }
    void prof_collect() {
        if (!prof_window && !prof_chunk) return;
        std::lock_guard<std::mutex> lk(prof_mu);
        int prev_dev = 0;
        cudaGetDevice(&prev_dev);
        std::vector<std::pair<const Dev*, cudaEvent_t>> last;
        for (const ProfMark& m : prof_marks) {
            auto it = std::find_if(last.begin(), last.end(), [&](const auto& x) { return x.first == m.d; });
            if (it != last.end()) {
                float ms = 0.0f;
                cudaSetDevice(m.d->id);   // (a pair of the same device's events, read on that device)
                if (cudaEventElapsedTime(&ms, it->second, m.ev) == cudaSuccess)
                    (prof_chunk ? cprof_ms : st.prof_ms)[m.stage] += ms;
                it->second = m.ev;
            } else {
                last.push_back({m.d, m.ev});
            }
        }
        cudaSetDevice(prev_dev);
        for (const ProfMark& m : prof_marks) prof_pool[(size_t) m.d->id % prof_pool.size()].push_back(m.ev);
        prof_marks.clear();
        if (!prof_chunk) ++st.prof_tokens;
    }
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
        std::vector<float> v((size_t) T * g.hc * g.dim);
        cudaSetDevice(d.id);
        cudaMemcpyAsync(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        std::lock_guard<std::mutex> lk(dump_mu);
        const int64_t hdr[3] = {il, p0, T};
        std::fwrite(hdr, 8, 3, dump_l);
        std::fwrite(v.data(), 4, v.size(), dump_l);
        std::fflush(dump_l);
    }

    ~Impl() {
        eg_io.stop();   // (its threads write into h_eg)
        for (auto& w : workers) w->stop();
        for (size_t i = 0; i < chunks.size(); ++i)
            if (chunks[i] && chunks[i]->blas) {
                cudaSetDevice(devs[i].id);
                cublasDestroy(chunks[i]->blas);
            }
        quit.store(true);
        if (service.joinable()) service.join();
        {
            std::lock_guard<std::mutex> lk(pf_mu);
        }
        pf_cv.notify_all();
        if (pf_th.joinable()) pf_th.join();
        {
            std::lock_guard<std::mutex> lk(sw_mu);
        }
        sw_cv.notify_all();
        ram_cv.notify_all();
        for (std::thread& t : cache_ths)
            if (t.joinable()) t.join();
        if (ram_th.joinable()) ram_th.join();
        for (size_t i = 0; i < ram_chunks.size(); ++i) os::anon_free(ram_chunks[i], ram_chunk_bytes[i]);
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
#if defined(STRATA_DS4_MMQ)
            mmq::destroy(d.mmq);   // (the prompt path's MMQ scratch: cudaMalloc'd on first use, not in devs' allocs)
            d.mmq = nullptr;
#endif
            for (void* p : d.allocs) cudaFree(p);
            if (d.st) cudaStreamDestroy(d.st);
            if (d.ev) cudaEventDestroy(d.ev);
        }
        for (void* p : host_allocs) cudaFreeHost(p);
        for (Layer& l : L)
            for (uint8_t* b : l.host)
                if (b != nullptr) {
                    (void) b;   // host blobs live in one mapping per layer, unmapped below
                }
        for (auto& m : host_maps) {
            if (chunk > 0) cudaHostUnregister(m.first);
            os::anon_free(m.first, m.second);
        }
    }
    std::vector<std::pair<void*, size_t>> host_maps;

    template <class T>
    bool dalloc(Dev& d, T*& p, size_t n, std::string& err, size_t* acct = nullptr) {
        void* v = nullptr;
        cudaSetDevice(d.id);
        const size_t b = align256(n * sizeof(T));
        // the row-split prefill's replica arenas: a swap's dense weights are carved out of one cudaMalloc, and the
        // buffers an earlier swap put there are simply overwritten (the arena is the layer's home, not a leak)
        if (d.bump != nullptr) {
            Bump& bp = *d.bump;
            if (bp.owner != d.id) {
                err = "internal: a CUDA" + std::to_string(bp.owner) + " replica arena used on CUDA" +
                      std::to_string(d.id);
                return false;
            }
            if (bp.off + b > bp.bytes) {
                err = "the row-split replica arena on CUDA" + std::to_string(d.id) + " is short by " +
                      std::to_string((bp.off + b - bp.bytes) >> 20) + " MiB (a layer larger than its slot)";
                return false;
            }
            v = bp.base + bp.off;
            bp.off += b;   // (the arena was zeroed once when it was made: every carve is fully written by its upload)
            if (acct != nullptr) *acct += b;
            p = (T*) v;
            return true;
        }
        const cudaError_t mae = cudaMalloc(&v, b);
        if (mae != cudaSuccess) {
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            err = "cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed: " +
                  cudaGetErrorString(mae) + " (free " + std::to_string(fr >> 20) + " of " + std::to_string(tot >> 20) +
                  " MiB)";
            return false;
        }
        // zeroed before anything lands in it: the loads copy in on the default stream, which the non-blocking d.st
        // does not order against - a zeroing still queued there could wipe the weights just copied
        if (cudaMemsetAsync(v, 0, b, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            cudaFree(v);
            err = "zeroing " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        d.allocs.push_back(v);
        if (acct) *acct += b;
        else d.scratch += b;
        p = (T*) v;
        return true;
    }

    // the prompt arena's carve: dalloc's device part, out of the device's one arena instead of a cudaMalloc.
    // The sizing pass (no arena yet) only adds the bytes; the carving pass takes them, zeroed, at 256-byte boundaries.
    // fixed/per_row (elements) say what a prompt of T rows touches of the buffer: fixed + per_row*T - a row-major
    // buffer passes its per-token stride, one every prompt uses whole leaves the default (fixed = (size_t)-1: all).
    template <class T>
    bool carve(Dev& d, T*& p, size_t n, std::string& err, size_t fixed = (size_t) -1, size_t per_row = 0) {
        const size_t b = align256(n * sizeof(T));
        if (d.arena == nullptr) {   // (sizing: the arena is cudaMalloc'd to exactly this many bytes)
            d.arena_bytes += b;
            p = nullptr;
            return true;
        }
        if (d.arena_off + b > d.arena_bytes) {
            err = "internal: CUDA" + std::to_string(d.id) + "'s prompt arena is short by " +
                  std::to_string(((d.arena_off + b) - d.arena_bytes) >> 20) + " MiB";
            return false;
        }
        void* v = d.arena + d.arena_off;
        if (fixed == (size_t) -1)   // the whole buffer, at any row count
            d.arena_spans.push_back({d.arena_off, b, b, 0});
        else
            d.arena_spans.push_back({d.arena_off, b, align256(std::min(n, fixed) * sizeof(T)), per_row * sizeof(T)});
        d.arena_off += b;
        cudaSetDevice(d.id);
        if (cudaMemsetAsync(v, 0, b, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            err = "zeroing " + std::to_string(b >> 20) + " MiB of CUDA" + std::to_string(d.id) + "'s prompt arena failed";
            return false;
        }
        d.scratch += b;
        p = (T*) v;
        return true;
    }
    static Layer& member(Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    static const Layer& member(const Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    // the Layer a chunk runs at il: the split's own, or (the row-split prefill) the replica holding it on the chunk's
    // device - every flow call site goes through this, the published state with it
    Layer& lof(Chunk& c, int il) {
        Layer* p = il >= 0 && il < (int) c.lay_override.size() ? c.lay_override[(size_t) il] : nullptr;
        return p != nullptr ? *p : L[(size_t) il];
    }
    const Layer& lof(const Chunk& c, int il) const {
        const Layer* p = il >= 0 && il < (int) c.lay_override.size() ? c.lay_override[(size_t) il] : nullptr;
        return p != nullptr ? *p : L[(size_t) il];
    }
    // bytes of a VRAM slot of l's experts (a member's part of each, in a group)
    static size_t unit(const Layer& l) { return l.nm > 1 ? l.hblob : l.blob; }
    // the members of a group of nm that take heads: the most the heads and the wo_a groups split among evenly (64
    // heads and 8 groups: 9 -> 8, 7 or 6 -> 4, 5 -> 4, 3 -> 2); the first members the rest, only their FFN share
    int head_members(int nm) const {
        int na = nm;
        while (na > 1 && (g.heads % na != 0 || g.o_groups % na != 0)) --na;
        return na;
    }
    // slot s of member x: [0, n_perm) in its own bank (vram_base), [n_perm, n_perm + n_lend) in the prompt arena
    // (lend_base, between prompts only - see enter_prefill / exit_prefill)
    static uint8_t* slot_ptr(const Layer& x, int s) {
        return s < x.n_perm ? x.vram_base + (size_t) s * unit(x)
                            : x.lend_base + (size_t) (s - x.n_perm) * unit(x);
    }
    // the layer's CPU lane reads its experts in place from the model's files: an expert a prompt evicts from the
    // arena can go back there (a host-copy lane cannot: the RAM it holds is not the file's pages)
    bool lane_file(const Layer& l) const { return l.lay.fmt == gg::kFmtMXFP4 || file_experts; }
    // --ep-prefill, or DS4_EP=1 (the checks): the prompt streamed and computed by every device, a share of each
    // layer's experts apiece
    bool ep_do() const {   // run expert-parallel (--ep-prefill, or DS4_EP=1 in the checks)
        if (o.ep_prefill) return true;
        const char* e = std::getenv("DS4_EP");
        return e != nullptr && std::atoi(e) != 0;
    }
    bool ep_on() const {   // allocate its staging (DS4_EP_ALLOC: the check's arm that keeps the placement equal)
        if (ep_do()) return true;
        const char* e = std::getenv("DS4_EP_ALLOC");
        return e != nullptr && std::atoi(e) != 0;
    }
    // DS4_PASS_LEND=1: the prompt's two passes lend each other's expert banks (see pl_borrow).  Off by default, and
    // only under DS4_TAIL_PREFILL: the banks of a decoder-tail layer and of an encoder layer on the same device are
    // allocated as one block when the model loads, so either layer can use the whole block in the pass it runs
    static bool pl_do() {
        const char* e = std::getenv("DS4_PASS_LEND");
        return e != nullptr && std::atoi(e) != 0;
    }
    // --row-split-prefill, or DS4_ROW_SPLIT=1 (the checks): every device holds the same layer group and runs its own
    // chunk of the prompt's rows through it, group by group.  The plan (K layers a group, G groups) is computed at
    // load from the room the expert slots leave; DS4_ROW_SPLIT_K / _BUDGET_MB / _FRAC / _FORCE are its knobs.
    bool rp_do() const {
        if (o.row_split_prefill) return true;
        const char* e = std::getenv("DS4_ROW_SPLIT");
        return e != nullptr && std::atoi(e) != 0;
    }
    static int rp_env_int(const char* n, int dflt) {
        const char* e = std::getenv(n);
        return e != nullptr && *e != 0 ? std::atoi(e) : dflt;
    }
    static double rp_env_f(const char* n, double dflt) {
        const char* e = std::getenv(n);
        return e != nullptr && *e != 0 ? std::atof(e) : dflt;
    }
    // --chunk-mmq, or DS4_MMQ=1 (the checks): the chunked prompt's MXFP4 expert sets through the MMQ kernels
    // (ds4_mmq.cu), a window of DS4_MMQ_ROWS activation rows at a time (1024)
    bool mmq_do() const {
        if (o.chunk_mmq) return true;
        const char* e = std::getenv("DS4_MMQ");
        return e != nullptr && std::atoi(e) != 0;
    }
    static int mmq_rows() {
        const char* e = std::getenv("DS4_MMQ_ROWS");
        const int n = e != nullptr ? std::atoi(e) : 0;
        return n > 0 ? n : 1024;
    }
    // host-mapped memory (the same address on the host and every device), zeroed
    template <class T>
    bool hmap(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "host-mapped memory (" + std::to_string((n * sizeof(T)) >> 20) + " MiB)";
            return false;
        }
        std::memset(v, 0, n * sizeof(T) + 64);
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocPortable) != cudaSuccess) {
            err = "cudaHostAlloc " + std::to_string((n * sizeof(T)) >> 20) + " MiB failed";
            return false;
        }
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }

    // ---- weights
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
        std::vector<uint8_t> tmp(t.bytes);
        if (!pk.read(t, tmp.data())) {
            err = "read failed";
            return false;
        }
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dst, tmp.data(), t.bytes, cudaMemcpyHostToDevice));
        return true;
    }
    // a GGUF's BF16 or Q8_0 matrix ([out][in]) onto the device, as mat() takes an EXL3 one: whole, or part `part` of its
    // columns (outputs: a range of its rows) or of its rows (inputs: a range of each row, whole Q8_0 blocks) - a
    // standalone [n][k] matrix
    bool mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct, int part,
                  int np, bool cols) {
        const bool q8 = t.dtype == "Q8_0", q6 = t.dtype == "Q6_K";
        if (t.shape.size() != 2 || (q8 && t.shape[1] % 32 != 0) || (q6 && t.shape[1] % 256 != 0)) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        // a row's bytes, and the bytes of v of its values (BF16: 2 each; Q8_0: 34 every 32; Q6_K: 210 every 256,
        // decoded to BF16 here - a block's scales interleave its codes, so a part cannot take the file's slice)
        auto bytes_of = [&](size_t v) { return q8 ? v / 32 * gg::kQ8Bytes : v * 2; };
        std::vector<uint8_t> w;
        if (q6) {
            constexpr size_t kB6 = 210;
            std::vector<uint8_t> raw((size_t) n * (size_t) (k / 256) * kB6);
            if (t.bytes != raw.size() || !pk.read(t, raw.data())) {
                err = pre + ": read failed";
                return false;
            }
            w.resize((size_t) n * bytes_of((size_t) k));
            for (int r = 0; r < n; ++r) {
                const uint8_t* b = raw.data() + (size_t) r * (size_t) (k / 256) * kB6;
                uint16_t* y = (uint16_t*) (w.data() + (size_t) r * bytes_of((size_t) k));
                for (int blk = 0; blk < k / 256; ++blk, b += kB6, y += 256) {
                    const uint8_t* ql = b, *qh = b + 128;        // ggml's block_q6_K: ql, qh, the 16 int8 scales
                    const int8_t* sc = (const int8_t*) (b + 192);
                    const float dd = f16_to_f32((uint16_t) (b[208] | b[209] << 8));
                    for (int half = 0; half < 2; ++half) {       // 128 values a half (ggml's dequantize_row_q6_K)
                        const uint8_t* l0 = ql + 64 * half, *h0 = qh + 32 * half;
                        const int8_t* s0 = sc + 8 * half;
                        uint16_t* yy = y + 128 * half;
                        for (int l = 0; l < 32; ++l) {
                            const int is = l / 16;
                            const int q1 = (int) ((l0[l] & 0xF) | (((h0[l] >> 0) & 3) << 4)) - 32;
                            const int q2 = (int) ((l0[l + 32] & 0xF) | (((h0[l] >> 2) & 3) << 4)) - 32;
                            const int q3 = (int) ((l0[l] >> 4) | (((h0[l] >> 4) & 3) << 4)) - 32;
                            const int q4 = (int) ((l0[l + 32] >> 4) | (((h0[l] >> 6) & 3) << 4)) - 32;
                            yy[l] = f32_to_bf16(dd * (float) s0[is + 0] * (float) q1);
                            yy[l + 32] = f32_to_bf16(dd * (float) s0[is + 2] * (float) q2);
                            yy[l + 64] = f32_to_bf16(dd * (float) s0[is + 4] * (float) q3);
                            yy[l + 96] = f32_to_bf16(dd * (float) s0[is + 6] * (float) q4);
                        }
                    }
                }
            }
        } else {
            w.resize((size_t) n * bytes_of((size_t) k));
            if (t.bytes != w.size() || !pk.read(t, w.data())) {
                err = pre + ": read failed";
                return false;
            }
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (q8 ? 32 : 8) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " +
                      std::to_string(np) + " parts";
                return false;
            }
            const size_t rb = bytes_of((size_t) k);
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = bytes_of((size_t) wd);
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r)
                    std::memcpy(p.data() + r * pb, w.data() + r * rb + bytes_of((size_t) part * wd), pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = q8 ? gg::kFmtQ8_0 : gg::kFmtBF16;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = q8 ? (int) bytes_of((size_t) k) : k;
        return true;
    }
    // a Q8_0 tensor's values as F32 (host)
    static std::vector<float> q8_values(const uint8_t* q, size_t n) {
        std::vector<float> v(n);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = q + b * gg::kQ8Bytes;
            const float d = f16_to_f32((uint16_t) (p[0] | p[1] << 8));
            for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return v;
    }
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    // (rot: split128's - an EXL3 matrix's uneven 128-block shares going round the members; a GGUF's BF16 / Q8_0 matrix
    // splits evenly, its groups only as wide as that goes)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0) {
        if (const Tensor* tw = pk.get(pre + ".weight");
            tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0" || tw->dtype == "Q6_K"))
            return mat_bf16(d, pre, *tw, m, err, acct, part, np, cols);
        const Tensor* tr = pk.get(pre + ".trellis");
        const Tensor* su = pk.get(pre + ".suh");
        const Tensor* sv = pk.get(pre + ".svh");
        if (tr == nullptr || su == nullptr || sv == nullptr || tr->shape.size() != 3) {
            err = pre + ": not an EXL3 matrix in the pack";
            return false;
        }
        if (pk.get(pre + ".mul1") == nullptr) {
            err = pre + ": only the mul1 codebook is covered";
            return false;
        }
        m.k = (int) tr->shape[0] * 16;
        m.n = (int) tr->shape[1] * 16;
        m.K = (int) tr->shape[2] / 16;
        m.cb = 2;
        m.ld = 0;
        if (np > 1) {
            const int whole = cols ? m.n : m.k;
            int at = 0, w = 0;
            if (whole % 128 != 0 || whole / 128 < np) {
                err = pre + ": " + std::to_string(whole) + (cols ? " columns" : " rows") + " do not split in " +
                      std::to_string(np) + " parts of 128-blocks";
                return false;
            }
            split128(whole, np, part, at, w, rot);
            std::vector<uint8_t> t0(tr->bytes), u0(su->bytes), v0(sv->bytes);
            if (!pk.read(*tr, t0.data()) || !pk.read(*su, u0.data()) || !pk.read(*sv, v0.data())) {
                err = pre + ": read failed";
                return false;
            }
            const size_t tile = tr->bytes / ((size_t) (m.k / 16) * (m.n / 16));
            std::vector<uint8_t> t1, u1, v1;
            if (cols) {   // the tiles [c0 / 16, (c0 + w) / 16) of each 16-row slice, svh [c0, c0 + w)
                const size_t c0 = (size_t) at, nt = (size_t) m.n / 16, nth = (size_t) w / 16;
                t1.resize((size_t) (m.k / 16) * nth * tile);
                for (size_t sl = 0; sl < (size_t) m.k / 16; ++sl)
                    std::memcpy(t1.data() + sl * nth * tile, t0.data() + (sl * nt + c0 / 16) * tile, nth * tile);
                u1 = u0;
                v1.assign(v0.begin() + (long) (c0 * 2), v0.begin() + (long) ((c0 + w) * 2));
                m.n = w;
            } else {      // the 16-row slices [r0 / 16, (r0 + w) / 16), suh [r0, r0 + w)
                const size_t r0 = (size_t) at, row = (size_t) (m.n / 16) * tile;
                t1.assign(t0.begin() + (long) (r0 / 16 * row), t0.begin() + (long) ((r0 + w) / 16 * row));
                u1.assign(u0.begin() + (long) (r0 * 2), u0.begin() + (long) ((r0 + w) * 2));
                v1 = v0;
                m.k = w;
            }
            uint32_t* dt = nullptr;
            uint16_t *ds = nullptr, *dv = nullptr;
            if (!dalloc(d, dt, t1.size() / 4, err, acct) || !dalloc(d, ds, u1.size() / 2, err, acct) ||
                !dalloc(d, dv, v1.size() / 2, err, acct))
                return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(dt, t1.data(), t1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, u1.data(), u1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, v1.data(), v1.size(), cudaMemcpyHostToDevice));
            m.trellis = dt;
            m.suh = ds;
            m.svh = dv;
            return x3::supported(m) || (err = pre + ": a part not covered by the EXL3 kernels", false);
        }
        uint32_t* dt = nullptr;
        uint16_t *ds = nullptr, *dv = nullptr;
        if (!dalloc(d, dt, tr->bytes / 4, err, acct) || !dalloc(d, ds, su->bytes / 2, err, acct) ||
            !dalloc(d, dv, sv->bytes / 2, err, acct))
            return false;
        if (!upload(d, *tr, dt, err) || !upload(d, *su, ds, err) || !upload(d, *sv, dv, err)) return false;
        m.trellis = dt;
        m.suh = ds;
        m.svh = dv;
        if (!x3::supported(m)) {
            err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " K " + std::to_string(m.K) +
                  " not covered by the EXL3 kernels";
            return false;
        }
        return true;
    }
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0) {
        const Tensor* t = pk.get(name);
        if (t == nullptr) {
            err = name + ": not in the pack";
            return false;
        }
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        size_t n = 0;
        std::vector<float> v;
        if (t->dtype == "F32") {
            n = t->bytes / 4;
            v.resize(n);
            std::memcpy(v.data(), raw.data(), t->bytes);
        } else if (t->dtype == "F16" || t->dtype == "BF16") {
            n = t->bytes / 2;
            v.resize(n);
            const uint16_t* s = (const uint16_t*) raw.data();
            for (size_t i = 0; i < n; ++i) v[i] = t->dtype == "F16" ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
        } else if (t->dtype == "Q8_0") {   // (a GGUF's router may come so)
            n = t->bytes / gg::kQ8Bytes * 32;
            v = q8_values(raw.data(), n);
        } else {
            err = name + ": dtype " + t->dtype + " is not a float";
            return false;
        }
        if (want != 0 && n != want) {
            err = name + ": " + std::to_string(n) + " values, expected " + std::to_string(want);
            return false;
        }
        if (!dalloc(d, p, n, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
        return true;
    }
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
        const Tensor* t = pk.get(name);
        if (t != nullptr && (t->dtype == "F32" || t->dtype == "BF16" || t->dtype == "Q8_0")) {   // (a GGUF's)
            const size_t n = t->dtype == "Q8_0" ? t->bytes / gg::kQ8Bytes * 32 : t->bytes / (t->dtype == "F32" ? 4 : 2);
            std::vector<uint8_t> raw(t->bytes);
            if (!pk.read(*t, raw.data())) {
                err = name + ": read failed";
                return false;
            }
            const std::vector<float> q = t->dtype == "Q8_0" ? q8_values(raw.data(), n) : std::vector<float>();
            std::vector<uint16_t> h(n);
            for (size_t i = 0; i < n; ++i) {
                float v;
                if (t->dtype == "F32") std::memcpy(&v, raw.data() + i * 4, 4);
                else if (t->dtype == "Q8_0") v = q[i];
                else v = bf16_to_f32(((const uint16_t*) raw.data())[i]);
                h[i] = strata::kernels::f16_from_f32(v);
            }
            if (!dalloc(d, p, n, err, acct)) return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
            return true;
        }
        if (t == nullptr || t->dtype != "F16") {
            err = name + ": not an F16 tensor in the pack";
            return false;
        }
        if (!dalloc(d, p, t->bytes / 2, err, acct)) return false;
        return upload(d, *t, p, err);
    }

    size_t mat_bytes(const std::string& pre) const {
        if (const Tensor* t = pk.get(pre + ".weight")) {
            if (t->dtype == "Q6_K") {   // (mat_bf16 decodes it to BF16 at load: 2 bytes a value)
                size_t v = 1;
                for (auto d : t->shape) v *= (size_t) d;
                return align256(v * 2);
            }
            return align256(t->bytes);
        }
        size_t b = 0;
        for (const char* s : {".trellis", ".suh", ".svh"})
            if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
        return b;
    }
    size_t tensor_bytes(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return t ? align256(t->bytes * (t->dtype == "F32" ? 1 : 2)) : 0;
    }

    // ---- expert regions and blob layout
    // a GGUF's MXFP4 experts: the blob is gate | up | down ([out][in / 32] blocks each, as the file holds them), every
    // expert's matrix r at the same stride from expert 0's in one file (the GGUF's [n_exp][out][in] tensor)
    // (the other block types: lay.fmt stays kFmtMXFP4 - read in place as these - and lay.K[r] is role r's ggml type)
    static int gg_type_of(const std::string& dtype) {
        for (int t : {gg::kTypeMXFP4, 10, 11, 12, 16, 17, 18, 21, 22, 23})
            if (dtype == gg::type_name(t)) return t;
        return -1;
    }
    bool expert_layout_mx(Layer& l, std::string& err) {
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
            const Tensor* t0 = pk.get(P + "0." + parts[r] + ".weight");
            const Tensor* t1 = l.n_exp > 1 ? pk.get(P + "1." + parts[r] + ".weight") : t0;
            const int ty = t0 != nullptr ? gg_type_of(t0->dtype) : -1;
            const size_t bytes = ty > 0 ? (size_t) n * gg::type_row_bytes(ty, k) : 0;
            if (t0 == nullptr || t1 == nullptr || bytes == 0 || t0->bytes != bytes) {
                err = P + "0." + parts[r] + ": not an MXFP4 or low-bit [" + std::to_string(n) + "][" + std::to_string(k) +
                      "] matrix the engine reads (" + (t0 != nullptr ? t0->dtype : std::string("missing")) + ")";
                return false;
            }
            l.mx_file[r] = t0->file;
            l.mx_off[r] = t0->off;
            l.mx_stride[r] = l.n_exp > 1 ? t1->off - t0->off : bytes;
            for (int e = 0; e < l.n_exp; ++e) {
                const Tensor* t = pk.get(P + std::to_string(e) + "." + parts[r] + ".weight");
                if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes || t->file != l.mx_file[r] ||
                    t->off != l.mx_off[r] + (uint64_t) e * l.mx_stride[r]) {
                    err = P + std::to_string(e) + "." + parts[r] + ": missing, or not where the layer's others are";
                    return false;
                }
            }
            l.lay.trellis[r] = off;
            l.lay.suh[r] = l.lay.svh[r] = off;
            l.lay.K[r] = ty;
            off += align256(bytes);
        }
        l.lay.fmt = gg::kFmtMXFP4;
        if (!gg::mx_experts(l.lay) && gg::expert_type(l.lay, 0) != gg::expert_type(l.lay, 1)) {
            err = P + ": gate / up / down of types " + gg::type_name(l.lay.K[0]) + " / " + gg::type_name(l.lay.K[1]) +
                  " / " + gg::type_name(l.lay.K[2]) + " (gate and up must be the same type)";
            return false;
        }
        l.blob = off;
        return true;
    }
    const uint8_t* mx_mat(const Layer& l, int e, int r) const {   // expert e's matrix r: its RAM-tier slot, else the file's
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0)   // (a copy with NH > 1: part 0's rows only - whole: read_mx)
            return copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] + cp_lay[(size_t) l.il][(size_t) r]
                             : ram_slots[(size_t) l.ram[(size_t) e]] + l.lay.trellis[r];
        return pk.at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r]);
    }
    size_t mx_part_bytes(const Layer& l, int r) const {   // (its type's)
        return r < 2 ? (size_t) g.ff * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                     : (size_t) g.dim * gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff);
    }
    // (copy_mode) matrix r's rows, and the rows of one NUMA part of it
    int mx_n(int r) const { return r < 2 ? g.ff : g.dim; }
    size_t mx_row_b(const Layer& l, int r) const { return mx_part_bytes(l, r) / (size_t) mx_n(r); }
    // part X's rows [X n / NH, ..) of expert e's matrix r: its copy's part on node X, else where the file has them
    const uint8_t* mx_part_ptr(const Layer& l, int e, int r, int X) const {
        if (copy_mode && l.ram[(size_t) e] >= 0)
            return ram_part[(size_t) l.ram[(size_t) e]][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
        return mx_mat(l, e, r) + (size_t) X * (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
    }
    // a copy's NH parts -> the blob, and back (each matrix's rows of part X at X n / NH rows into the blob's matrix)
    void ram_to_blob(const Layer& l, int s, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.trellis[r] + (size_t) X * nb, ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], nb);
        }
    }
    void blob_to_ram(const Layer& l, const uint8_t* blob, int s) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], blob + l.lay.trellis[r] + (size_t) X * nb, nb);
        }
    }
    // expert e (from the file) -> its blob; and member m of nm's part of a blob (gate / up rows [m n_ff / nm, ..), down
    // inputs of the same range: in each row the blocks [m n_ff / nm / 32, ..)) and back
    bool read_mx(const Layer& l, int e, uint8_t* blob) const {   // (from its RAM-tier slot or copy when it has one)
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) {
            if (copy_mode) ram_to_blob(l, l.ram[(size_t) e], blob);
            else std::memcpy(blob, ram_slots[(size_t) l.ram[(size_t) e]], l.blob);
            return true;
        }
        return read_mx_file(l, e, blob);
    }
    bool read_mx_file(const Layer& l, int e, uint8_t* blob) const {   // (the movers: never a slot that may change)
        for (int r = 0; r < 3; ++r)
            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r], mx_part_bytes(l, r),
                            blob + l.lay.trellis[r]))
                return false;
        return true;
    }
    void mx_blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        const int ffh = g.ff / l.nm;
        const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                     dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                     dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
        for (int r = 0; r < 2; ++r)
            std::memcpy(dst + l.hlay.trellis[r], blob + l.lay.trellis[r] + (size_t) m * ffh * gr, (size_t) ffh * gr);
        for (size_t row = 0; row < (size_t) g.dim; ++row)
            std::memcpy(dst + l.hlay.trellis[2] + row * dh, blob + l.lay.trellis[2] + row * dr + (size_t) m * dh, dh);
    }
    void mx_half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        const int ffh = g.ff / l.nm;
        const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                     dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                     dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
        for (int r = 0; r < 2; ++r)
            std::memcpy(blob + l.lay.trellis[r] + (size_t) m * ffh * gr, src + l.hlay.trellis[r], (size_t) ffh * gr);
        for (size_t row = 0; row < (size_t) g.dim; ++row)
            std::memcpy(blob + l.lay.trellis[2] + row * dr + (size_t) m * dh, src + l.hlay.trellis[2] + row * dh, dh);
    }

    bool expert_layout(Layer& l, std::string& err) {
        if (const Tensor* t = pk.get(l.name + "ffn.experts.0.w1.weight"); t != nullptr && gg_type_of(t->dtype) > 0)
            return expert_layout_mx(l, err);
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.region_off.resize((size_t) l.n_exp);
        l.region_bytes.resize((size_t) l.n_exp);
        l.region_file.resize((size_t) l.n_exp);
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const Tensor* tr = pk.get(P + "0." + parts[r] + ".trellis");
            if (tr == nullptr || pk.get(P + "0." + parts[r] + ".mul1") == nullptr) {
                err = P + "0." + parts[r] + ": not an EXL3 (mul1) matrix";
                return false;
            }
            l.lay.K[r] = (int) tr->shape[2] / 16;
            l.lay.trellis[r] = off;
            off += align256(tr->bytes);
            l.lay.suh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".suh")->bytes);
            l.lay.svh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".svh")->bytes);
        }
        l.lay.cb = 2;
        l.blob = off;
        for (int e = 0; e < l.n_exp; ++e) {
            const std::string E = P + std::to_string(e) + ".";
            uint64_t lo = UINT64_MAX, hi = 0;
            int file = -1;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + parts[r] + pieces[q]);
                    if (t == nullptr) {
                        err = E + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    if (file < 0) file = t->file;
                    if (t->file != file) {
                        err = E + ": split across files";
                        return false;
                    }
                    lo = std::min(lo, t->off);
                    hi = std::max(hi, t->off + t->bytes);
                    if (e == 0) l.piece_bytes[r][q] = t->bytes;
                    else if (t->bytes != l.piece_bytes[r][q]) {
                        err = E + parts[r] + ": not the shape of expert 0's";
                        return false;
                    }
                }
            l.region_off[(size_t) e] = lo;
            l.region_bytes[(size_t) e] = hi - lo;
            l.region_file[(size_t) e] = file;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const size_t src = pk.get(E + parts[r] + pieces[q])->off - lo;
                    if (e == 0) l.piece_src[r][q] = src;
                    else if (src != l.piece_src[r][q]) {
                        err = E + ": the pieces lie differently from expert 0's";
                        return false;
                    }
                }
        }
        return true;
    }
    // the region of expert e (read from the file) -> its blob
    void repack(const Layer& l, const uint8_t* region, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t dst[3] = {l.lay.trellis[r], l.lay.suh[r], l.lay.svh[r]};
            for (int q = 0; q < 3; ++q) std::memcpy(blob + dst[q], region + l.piece_src[r][q], l.piece_bytes[r][q]);
        }
    }

    // the region of expert e -> its CPU-lane part X (columns [X n / NH, (X + 1) n / NH) of each matrix)
    void repack_part(const Layer& l, const uint8_t* region, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = region + l.piece_src[r][0];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], region + l.piece_src[r][1], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], region + l.piece_src[r][2] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }

    // ---- loading
    bool load(const Options& opt, std::string& err) {
        o = opt;
        const double t0 = now_ms();
        if (!pk.load(o.pack, err)) return false;
        g.from(pk);
        if (g.dim == 0 || g.nl == 0 || (int) g.ratios.size() < g.nl) {
            err = o.pack + "/ds4.txt: incomplete geometry";
            return false;
        }
        if (g.hd != 512 || g.ix_hd != 128 || g.k_exp > 8 || g.hc > 8) {
            err = "geometry not covered (head_dim " + std::to_string(g.hd) + ", index_head_dim " +
                  std::to_string(g.ix_hd) + ", experts per token " + std::to_string(g.k_exp) + ")";
            return false;
        }
        freqs_win = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow);
        freqs_comp = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow);

        // devices
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < ndev; ++i) ids.push_back(i);
        devs.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            Dev& d = devs[i];
            d.id = ids[i];
            CK(cudaSetDevice(d.id));
            CK(cudaStreamCreateWithFlags(&d.st, std::getenv("DS4_RP_CHECK") ? cudaStreamDefault   // (temporary: the check's streams blocking, so CUDA_LAUNCH_BLOCKING names each launch)
                                                   : cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
            {
                char bus[32] = "";
                if (cudaDeviceGetPCIBusId(bus, sizeof bus, d.id) == cudaSuccess) d.pci = bus;
            }
            if (!x3::prepare(d.st) || (pk.gguf && !gg::prepare(d.st))) {
                err = "exl3 scratch on CUDA" + std::to_string(d.id);
                return false;
            }
            CK(cudaMemGetInfo(&d.free0, &d.total));
        }
        live_dev = std::make_unique<LiveDev[]>(devs.size());
        // the VRAM each GPU leaves free: --reserve-mib (every GPU) wins, else --reserve-pct (one entry per GPU in the
        // split's order, a single entry for all, a short list repeating its last one; never under 64 MiB), else the
        // 600 MiB default.  Resolved once here: the load's decisions and place_experts all read it per device.
        resv.assign(devs.size(), 0);
        for (size_t i = 0; i < devs.size(); ++i) {
            size_t b = (size_t) o.reserve_mib << 20;
            if (!o.reserve_set && !o.reserve_pct.empty()) {
                const int p = std::max(0, o.reserve_pct[std::min(i, o.reserve_pct.size() - 1)]);
                b = (size_t) std::max(64, (int) ((double) devs[i].total * (double) p / 100.0));
            }
            resv[i] = std::min(b, devs[i].total);
        }
        // the groups' width: o.tp, less while the experts' blocks do not split in it (a member takes n_ff / tp of each
        // expert: the down's inputs in whole blocks - a GGUF's 256-value types split n_ff 2304 in 1, 3 or 9 parts)
        int tp = o.tp;
        if (pk.gguf && tp > 1) {
            int blk = 32;
            for (const auto& kv : pk.t) {
                const std::string& n = kv.first;
                if (n.find(".ffn.experts.") == std::string::npos || n.size() < 9 ||
                    n.compare(n.size() - 9, 9, "w2.weight") != 0)
                    continue;
                const int ty = gg_type_of(kv.second.dtype);
                if (ty >= 0) blk = std::max(blk, gg::type_block(ty));
            }
            while (tp > 1 && (g.ff % tp != 0 || (g.ff / tp) % blk != 0 || (g.ff / tp) % 128 != 0 || g.heads % tp != 0 ||
                              g.o_groups % tp != 0))
                --tp;
            if (tp != o.tp)
                std::fprintf(stderr, "ds4: tensor parallel %d -> %d: the experts' %d-value blocks split n_ff %d no other way "
                                     "(each GPU its own layers)\n", o.tp, tp, blk, g.ff);
        }
        // the groups: runs of tp devices of one kind (their memory), a device unlike the next alone
        // - or, tp the number of GPUs, one group of all of them whatever their kind (its tightest member sets the
        // experts' room); threads mode for the windows (decode) where a group is 2 wide or more or --threads on, the
        // prompt path (its own thread a GPU) as asked for - also where the experts' blocks took the groups' width down
        // to 1 (a 256-value type's: Maya's quants), which left every prompt in windows of 16
        par = o.threads || tp > 1;
        prompt_par = par || o.tp > 1;
        const bool tp_all = tp > 1 && tp == (int) devs.size();
        for (size_t i = 0; i < devs.size();) {
            Group G;
            G.mem.push_back((int) i);
            size_t j = i + 1;
            while ((int) G.mem.size() < tp && j < devs.size() && (tp_all || devs[j].total == devs[i].total))
                G.mem.push_back((int) j++);
            if ((int) G.mem.size() < tp && G.mem.size() > 1) {   // a short run at the end: devices of their own
                for (int m : G.mem) {
                    Group one;
                    one.mem.push_back(m);
                    groups.push_back(one);
                }
            } else groups.push_back(G);
            i = j;
        }
        for (size_t gi = 0; gi < groups.size(); ++gi)
            for (size_t m = 0; m < groups[gi].mem.size(); ++m) {
                devs[(size_t) groups[gi].mem[m]].grp = (int) gi;
                devs[(size_t) groups[gi].mem[m]].mi = (int) m;
            }
        // the prompt chunk's parts: --prefill-chunk-max (DS4_CHUNK_MAX), one ceiling per group in the split's order
        // (0 or absent: none).  It enables the chunked prompt path on its own - decode stays windows of serial - and
        // --prefill-chunk stays the whole split's ceiling; --prefill-chunk 0 still turns the path off
        const char* chunk_env = std::getenv("DS4_CHUNK");
        const char* cmax_env = std::getenv("DS4_CHUNK_MAX");
        const bool caps_given = !o.prefill_chunk_max.empty() || cmax_env != nullptr;
        std::vector<int> chunk_caps = o.prefill_chunk_max;
        if (cmax_env != nullptr) {
            chunk_caps.clear();
            for (const char* q = cmax_env; *q;) {
                chunk_caps.push_back(std::max(0, std::atoi(q)));
                while (*q && *q != ',') ++q;
                if (*q == ',') ++q;
            }
        }
        while (chunk_caps.size() < groups.size()) chunk_caps.push_back(0);
        if (chunk_caps.size() > groups.size()) {
            std::fprintf(stderr, "ds4: --prefill-chunk-max: %zu entries for %zu parts, the extras ignored\n",
                         chunk_caps.size(), groups.size());
            chunk_caps.resize(groups.size());
        }
        const bool chunk_off = o.prefill_chunk == 0 && chunk_env == nullptr;
        const bool chunk_on = !chunk_off && (prompt_par || o.prefill_chunk > 0 || caps_given);
        // whether the CPU lane's experts are read in place (--mmap-experts; auto: the pack's expert bytes beyond what the
        // GPUs can take - their free memory less the other weights and the prompt path's buffers - do not fit the RAM
        // this process can get beside kRamMargin), decided here because it sets the prompt chunk's default: in place
        // every chunk streams every expert VRAM does not hold from the files, so the larger the chunk the fewer times
        if (pk.gguf) file_experts = true;   // (a GGUF's: the files stay their source - copy_experts copies them into RAM
                                            // after VRAM's are in, by --mmap-experts)
        else if (o.mmap_experts >= 0) file_experts = o.mmap_experts > 0;
        else {
            double exp_b = 0, other_b = 0, room = 0;
            for (const auto& kt : pk.t) {
                const std::string& n = kt.first;
                if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                    (!o.dspark && n.rfind("mtp.", 0) == 0))
                    continue;
                (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
            }
            for (size_t i = 0; i < devs.size(); ++i)
                room += std::max(0.0, (double) devs[i].free0 - (double) resv[i] - (chunk_on ? 1.5 : 0.4) * (1 << 30)) *
                        o.vram_experts;
            file_experts = exp_b - std::max(0.0, room - other_b) + (double) kRamMargin > (double) ram_avail();
        }
        chunk = chunk_on ? (o.prefill_chunk > 0 ? o.prefill_chunk : file_experts ? kVlRows : 2048) : 0;
        if (chunk_env != nullptr) chunk = std::max(0, std::atoi(chunk_env));
        if (const char* v = std::getenv("DS4_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(v));
        if (chunk_on && chunk_env == nullptr && o.prefill_chunk <= 0 && caps_given) {   // (the parts' ceilings shape
            int mx = 0;                                                                // the run: no 8192 default)
            for (int c : chunk_caps) mx = std::max(mx, c);
            if (mx > 0) chunk = mx;
        }
        chunk_steps.assign(groups.size(), 0);
        if (chunk > 0) {
            chunk = std::min(kMaxChunk, std::max(chunk_env != nullptr ? 16 : 256, chunk / 16 * 16));
            // each part's step: the largest whole multiple of the step of the part before it at most its ceiling.  The
            // hop copies a part's step out of its feeder's chunks, so a consumer's chunk is whole feeder chunks (which
            // is also what lets the head read a step larger than the feeder's a piece at a time).  A ceiling that cannot
            // hold one step of the part before it flattens the whole chain to the smallest, as GLM does by default
            chunk_steps.assign(groups.size(), chunk);
            bool any_cap = false;
            for (int c : chunk_caps) any_cap = any_cap || c > 0;
            if (any_cap) {
                bool flat = false;
                int prev = 0;
                for (size_t gi = 0; gi < chunk_steps.size(); ++gi) {
                    const int cap = chunk_caps[gi] > 0 ? std::min(chunk, std::max(256, chunk_caps[gi] / 16 * 16)) : chunk;
                    if (gi == 0) {
                        prev = cap;
                    } else if (cap < prev) {
                        flat = true;
                        break;
                    } else {
                        prev = cap / prev * prev;
                    }
                    chunk_steps[gi] = prev;
                }
                if (flat) {
                    int mn = chunk;
                    for (int c : chunk_caps)
                        if (c > 0) mn = std::min(mn, std::max(256, c / 16 * 16));
                    std::fill(chunk_steps.begin(), chunk_steps.end(), mn);
                    std::fprintf(stderr,
                                 "ds4: the parts' prompt chunks flattened to %d: a part's ceiling cannot hold a whole "
                                 "step of the part before it\n", mn);
                }
                if (groups.size() > 1) {
                    std::string sh;
                    for (size_t gi = 0; gi < groups.size(); ++gi)
                        sh += (gi ? ", " : "") + std::string("CUDA") + std::to_string(devs[(size_t) groups[gi].mem[0]].id) +
                              " " + std::to_string(chunk_steps[gi]);
                    std::fprintf(stderr, "ds4: the parts' prompt chunks: %s\n", sh.c_str());
                }
            }
            eg_cap = (size_t) *std::max_element(chunk_steps.begin(), chunk_steps.end());
            tap_cap = std::max(kWin, 128);
        } else   // (only as asked for: --prefill-chunk 0, or --tp 1 with --threads off)
            std::fprintf(stderr,
                         "ds4: prompts run in windows of %d tokens (%s): long prompts prefill far slower than in\n"
                         "     chunks (a 7-GPU RTX 3090 rig, Maya-S read in place: 8K / 32K tokens 115 / 99 tokens/s,\n"
                         "     461 / 1010 in chunks of %d) - %s\n",
                         kWin, o.prefill_chunk == 0 ? "--prefill-chunk 0" : "--tp 1 with --threads off", kVlRows,
                         o.prefill_chunk == 0 ? "leave --prefill-chunk out for chunks"
                                              : "leave --tp 1 out, or pass --threads on, for chunks");
        // the vision tower on the first GPU, before the split: the layers get what it leaves
        if (o.vision && !load_vision(err)) {   // (a tower that cannot load leaves the model reading text: said loudly,
            if (o.vision_only) return false;   // and the server is told pictures are off - info_lines)
            std::fprintf(stderr, "ds4: WARNING: the vision tower did not load (%s): pictures are OFF, the model reads "
                                 "text only\n", err.c_str());
            vision_err = err.empty() ? "unknown reason" : err;
            err.clear();
            vt.release();
            img_tok = -1;
        }
        if (o.vision_only) return true;

        // layers: roles and the sources they read
        if (!o.dspark) g.n_mtp = 0;
        L.resize((size_t) (g.nl + g.n_mtp));
        int last_src = -1;
        for (int il = 0; il < g.nl + g.n_mtp; ++il) {
            Layer& l = L[(size_t) il];
            l.il = il;
            l.mtp = il >= g.nl;
            l.name = l.mtp ? "mtp." + std::to_string(il - g.nl) + "." : "layers." + std::to_string(il) + ".";
            l.n_exp = l.mtp ? g.d_n_exp : g.n_exp;
            l.k_exp = l.mtp ? g.d_k : g.k_exp;
            l.ratio = g.ratios[(size_t) il];
            if (l.mtp) {   // window attention only: no compressor, indexer or Engram
                if (l.ratio != 0 || l.n_exp <= 0 || l.k_exp <= 0 || l.k_exp > kPicks) {
                    err = l.name + ": a DSpark block of compress ratio " + std::to_string(l.ratio) + ", " +
                          std::to_string(l.n_exp) + " experts / " + std::to_string(l.k_exp) + " is not covered";
                    return false;
                }
                if (!expert_layout(l, err)) return false;
                continue;
            }
            l.kv_src = std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end();
            l.ix_src = std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end();
            l.owns_k = l.kv_src;
            l.cand_src = il == g.cand_src;
            l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
            if (l.kv_src) {
                Src s;
                s.layer = il;
                s.ratio = l.ratio;
                s.cap = (o.max_context + l.ratio - 1) / l.ratio + 1;
                srcs.push_back(s);
                last_src = (int) srcs.size() - 1;
            }
            if (l.ratio > 0) {
                if (last_src < 0 || srcs[(size_t) last_src].ratio != l.ratio) {
                    err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(l.ratio) +
                          " without a KV source of that ratio before it";
                    return false;
                }
                l.src = last_src;
                l.ik_src = last_src;
            }
            auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
            if (eg != g.eg_layers.end()) l.eg = (int) (eg - g.eg_layers.begin());
            if (!expert_layout(l, err)) return false;
        }

        // the split: each layer's dense weights + its experts, the head on the last GPU
        for (Layer& l : L) {
            const std::string& P = l.name;
            size_t b = 0;
            if (l.mtp && l.il == g.nl) b += mat_bytes("mtp.0.main_proj");
            for (const char* m : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1",
                                  "ffn.shared_experts.w2", "ffn.shared_experts.w3", "attn.compressor.wkv",
                                  "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv"})
                b += mat_bytes(P + m);
            for (int gi = 0; gi < g.o_groups; ++gi) b += mat_bytes(P + "attn.wo_a.slice." + std::to_string(gi));
            b += tensor_bytes(P + "hc_attn_fn") + tensor_bytes(P + "hc_ffn_fn") + tensor_bytes(P + "ffn.gate.weight");
            b += (size_t) g.win * win_rb() + (2u << 20);
            l.dense_bytes = b;
        }
        const size_t head_b = mat_bytes("head") + (size_t) g.vocab * 4 + (16u << 20);
        // the split's planning margin stays the 600 MiB default whatever --reserve-pct says: the layer split must not
        // move with the lending (the same split lets this change be measured against the build before it)
        const size_t per_dev_fixed = (size_t) kSplitReserveMib << 20;
        const size_t kv_per_src_dev = 0;   // counted after the split
        (void) kv_per_src_dev;
        std::vector<double> avail(groups.size(), 0.0);
        double avail_sum = 0;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            for (int m : groups[gi].mem)
                avail[gi] += std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed - 400.0 * (1 << 20));
            avail_sum += avail[gi];
        }
        double need_sum = (double) head_b;
        for (const Layer& l : L) need_sum += (double) l.dense_bytes + (double) l.blob * l.n_exp;
        {
            // contiguous layer ranges, each group's share of the total by its free memory
            size_t d = 0;
            double cum = 0, target = avail[0] / avail_sum * need_sum;
            std::vector<Layer*> order;   // the drafter's blocks first (the first GPU), then the trunk's layers
            for (Layer& l : L)
                if (l.mtp) order.push_back(&l);
            for (Layer& l : L)
                if (!l.mtp) order.push_back(&l);
            for (Layer* lp : order) {
                Layer& l = *lp;
                const double lb = (double) l.dense_bytes + (double) l.blob * l.n_exp;
                if (cum + lb / 2 > target && d + 1 < groups.size()) {
                    ++d;
                    target += avail[d] / avail_sum * need_sum;
                }
                l.grp = (int) d;
                cum += lb;
            }
            // with the chunked prompt path: the contiguous split that streams the least into its busiest member (its
            // VRAM short of its layers' experts, per member) - a chunk's groups run as a pipeline, its slowest stage
            // sets the pace; a token's decode passes every layer either way, the experts in VRAM the same in all
            if (chunk > 0 && groups.size() > 1 && groups.size() <= 6 && std::getenv("DS4_SPLIT") == nullptr) {
                const int G = (int) groups.size(), NL = g.nl;
                std::vector<double> cap(groups.size());
                for (int gi = 0; gi < G; ++gi) {
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    const double cb = (double) chunk_vram_est(gi);   // (the part's chunk buffers, not a flat 1.1 GiB)
                    double mn = 1e300;
                    for (int m : groups[(size_t) gi].mem)
                        mn = std::min(mn, std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed -
                                                            400.0 * (1 << 20) - cb));
                    cap[(size_t) gi] = mn * nm;
                }
                for (const Layer& l : L)
                    if (l.mtp) cap[0] -= (double) l.dense_bytes + (double) l.blob * l.n_exp;
                cap[(size_t) G - 1] -= (double) head_b;
                std::vector<double> pe(NL + 1, 0.0), pd(NL + 1, 0.0);   // prefix sums: experts, dense
                for (int il = 0; il < NL; ++il) {
                    pe[(size_t) il + 1] = pe[(size_t) il] + (double) L[(size_t) il].blob * L[(size_t) il].n_exp;
                    pd[(size_t) il + 1] = pd[(size_t) il] + (double) L[(size_t) il].dense_bytes;
                }
                auto cost = [&](int gi, int a, int b) {   // per-member streaming of layers [a, b) on group gi
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    const double dense = (pd[(size_t) b] - pd[(size_t) a]) * (nm > 1 ? 1.3 : 1.0);
                    return std::max(0.0, pe[(size_t) b] - pe[(size_t) a] - (cap[(size_t) gi] - dense)) / nm;
                };
                // best[gi][b]: the least max cost of layers [0, b) over groups 0..gi (each at least one layer)
                std::vector<std::vector<double>> best((size_t) G, std::vector<double>((size_t) NL + 1, 1e300));
                std::vector<std::vector<int>> cut((size_t) G, std::vector<int>((size_t) NL + 1, 0));
                for (int b = 1; b <= NL; ++b) best[0][(size_t) b] = cost(0, 0, b);
                for (int gi = 1; gi < G; ++gi)
                    for (int b = gi + 1; b <= NL; ++b)
                        for (int a = gi; a < b; ++a) {
                            const double v = std::max(best[(size_t) gi - 1][(size_t) a], cost(gi, a, b));
                            if (v < best[(size_t) gi][(size_t) b]) {
                                best[(size_t) gi][(size_t) b] = v;
                                cut[(size_t) gi][(size_t) b] = a;
                            }
                        }
                int b = NL;
                for (int gi = G - 1; gi >= 0; --gi) {
                    const int a = gi > 0 ? cut[(size_t) gi][(size_t) b] : 0;
                    for (int il = a; il < b; ++il) L[(size_t) il].grp = gi;
                    b = a;
                }
            }
            // DS4_SPLIT=a,b,..: the trunk layers each group takes (the drafter stays with the first)
            if (const char* sp = std::getenv("DS4_SPLIT")) {
                std::vector<int> n;
                for (const char* q = sp; *q;) {
                    n.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
                int il = 0;
                for (size_t gi = 0; gi < n.size() && gi < groups.size(); ++gi)
                    for (int j = 0; j < n[gi] && il < g.nl; ++j) L[(size_t) il++].grp = (int) gi;
                for (; il < g.nl; ++il) L[(size_t) il].grp = (int) groups.size() - 1;
            }
            // DS4_SPLIT_TAIL=a,b,..: the decoder-tail layers (candidate source + 1 .. the last) each group takes, on top
            // of DS4_SPLIT's encoder range - the tail pass then runs on every group instead of the last one alone
            if (const char* sp = std::getenv("DS4_SPLIT_TAIL")) {
                std::vector<int> n;
                for (const char* q = sp; *q;) {
                    n.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
                int il = g.cand_src >= 0 ? g.cand_src + 1 : g.nl;
                for (size_t gi = 0; gi < n.size() && gi < groups.size(); ++gi)
                    for (int j = 0; j < n[gi] && il < g.nl; ++j) L[(size_t) il++].grp = (int) gi;
            }
            // (the row split: the drafter's blocks ride with the head - the last chunk's taps and the head step are one
            // device's work there, which is what rp_static asks for; the blocks are small, the split's balance holds)
            if (rp_do() && g.n_mtp > 0)
                for (int j = 0; j < g.n_mtp; ++j) L[(size_t) (g.nl + j)].grp = (int) groups.size() - 1;
            for (Layer& l : L) {
                l.dev = groups[(size_t) l.grp].mem[0];
                l.nm = l.mtp ? 1 : (int) groups[(size_t) l.grp].mem.size();   // (the drafter's blocks: on the lead alone)
            }
            head_dev = L[(size_t) g.nl - 1].dev;
            d_dev = g.n_mtp > 0 ? L[(size_t) g.nl].dev : 0;
        }
        // the groups' members: copies of each layer's device state, a member's part of an expert
        for (Layer& l : L) {
            l.ff0 = 0;
            l.ffn = g.ff;
            l.ah0 = l.og0 = 0;
            l.ahn = g.heads;
            l.ogn = g.o_groups;
            if (l.nm == 1) continue;
            const int na = head_members(l.nm);
            if (g.heads % na != 0 || g.o_groups % na != 0 || g.ff % 128 != 0 || g.ff / 128 < l.nm) {
                err = "tensor parallel " + std::to_string(l.nm) + ": the heads, output groups or FFN do not split";
                return false;
            }
            for (int m = 1; m < l.nm; ++m) {
                Layer c = l;
                c.mates.clear();
                c.mi = m;
                c.dev = groups[(size_t) l.grp].mem[(size_t) m];
                l.mates.push_back(std::move(c));
            }
            // each member's FFN columns and its part of an expert's layout: an EXL3 expert's in whole 128-blocks, as even
            // as they go (split128, the odd blocks going round the members layer by layer); a GGUF's in even shares (its
            // groups are only as wide as its experts' blocks split evenly: mx_blob_to_half's n_ff / nm)
            const bool gguf_x = l.lay.fmt == gg::kFmtMXFP4;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const int ha = m - (l.nm - na);   // (its place among the members with heads)
                x.ahn = ha < 0 ? 0 : g.heads / na;
                x.ah0 = ha < 0 ? 0 : ha * x.ahn;
                x.ogn = ha < 0 ? 0 : g.o_groups / na;
                x.og0 = ha < 0 ? 0 : ha * x.ogn;
                if (gguf_x) {
                    x.ffn = g.ff / l.nm;
                    x.ff0 = m * x.ffn;
                } else
                    split128(g.ff, l.nm, m, x.ff0, x.ffn, l.il % l.nm);   // (rot: the odd blocks go round the members)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : x.ffn, n = r < 2 ? x.ffn : g.dim;
                    x.hlay.K[r] = l.lay.K[r];
                    x.hlay.trellis[r] = off;
                    if (gguf_x) {   // gate / up rows, down inputs: [n][k / block] blocks
                        const int ty = gg::expert_type(l.lay, r);
                        if (k % gg::type_block(ty) != 0) {
                            err = "tensor parallel " + std::to_string(l.nm) + ": the experts' " + gg::type_name(ty) +
                                  " blocks do not split";
                            return false;
                        }
                        x.hlay.suh[r] = x.hlay.svh[r] = off;
                        off += align256((size_t) n * gg::type_row_bytes(ty, k));
                        continue;
                    }
                    off += align256(x3::trellis_bytes(k, n, l.lay.K[r]));
                    x.hlay.suh[r] = off;
                    off += align256((size_t) k * 2);
                    x.hlay.svh[r] = off;
                    off += align256((size_t) n * 2);
                }
                x.hlay.cb = l.lay.cb;
                x.hlay.fmt = l.lay.fmt;
                x.hblob = off;
            }
        }
        // which devices hold each source's compressed KV / index keys
        for (Src& s : srcs) {
            s.ckv.assign(devs.size(), nullptr);
            s.ik.assign(devs.size(), nullptr);
            if (rp_do()) {   // the row-split prefill's copies: every device, its own rows (see forward_rowsplit)
                s.ckv_rp.assign(devs.size(), nullptr);
                s.ik_rp.assign(devs.size(), nullptr);
            }
        }
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                const Layer& l = member(l0, m);
                if (l.src >= 0) {
                    srcs[(size_t) l.src].ckv[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.src].layer].grp) srcs[(size_t) l.src].remote = true;
                }
                if (l.ix_src && l.ik_src >= 0) {
                    srcs[(size_t) l.ik_src].ik[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.ik_src].layer].grp) srcs[(size_t) l.ik_src].remote = true;
                }
            }

        // per device: the scratch, the caches, the dense weights
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            const int D = g.dim, H = g.hc;
            const size_t W = kWin, P = (size_t) kWin * kPicks, cap = (size_t) o.max_context + 64;
            bool ok = dalloc(d, d.h, W * H * D, err) && dalloc(d, d.h2, W * H * D, err) && dalloc(d, d.x, W * D, err) &&
                      dalloc(d, d.qr, W * g.q_lora, err) && dalloc(d, d.q, W * g.heads * g.hd, err) &&
                      dalloc(d, d.kvr, W * g.hd, err) && dalloc(d, d.ckr, W * g.hd, err) &&
                      dalloc(d, d.cgr, W * g.hd, err) && dalloc(d, d.latent, W * g.hd, err) &&
                      dalloc(d, d.ikr, W * g.ix_hd, err) && dalloc(d, d.iq, W * g.ix_heads * g.ix_hd, err) &&
                      dalloc(d, d.iw, W * 64, err) && dalloc(d, d.scores, W * cap, err) &&
                      dalloc(d, d.bm, W * bm_stride(), err) && dalloc(d, d.o, W * g.heads * g.hd, err) &&
                      dalloc(d, d.oa, W * g.o_groups * g.o_lora, err) && dalloc(d, d.ao, W * D, err) &&
                      dalloc(d, d.mix, W * 32, err) && dalloc(d, d.a_pre, W * 8, err) && dalloc(d, d.a_post, W * 8, err) &&
                      dalloc(d, d.a_comb, W * 64, err) && dalloc(d, d.f_pre, W * 8, err) &&
                      dalloc(d, d.f_post, W * 8, err) && dalloc(d, d.f_comb, W * 64, err) && dalloc(d, d.pre, W * 8, err) &&
                      dalloc(d, d.glog, W * 1024, err) && dalloc(d, d.glog_a, 2 * W * 1024, err) &&
                      dalloc(d, d.rw, W * kPicks, err) &&
                      dalloc(d, d.ridx, W * kPicks, err) && dalloc(d, d.moe_out, W * D, err) &&
                      dalloc(d, d.sg, W * g.ff, err) && dalloc(d, d.su, W * g.ff, err) && dalloc(d, d.sh, W * g.ff, err) &&
                      dalloc(d, d.sh_out, W * D, err) && dalloc(d, d.cpu_part, W * D, err) &&
                      dalloc(d, d.freqs_win, 64, err) && dalloc(d, d.freqs_comp, 64, err) &&
                      dalloc(d, d.topk, W * topk_stride(), err) && dalloc(d, d.cand, W * cand_stride(), err) &&
                      dalloc(d, d.tok, 16, err) && dalloc(d, d.emb_rows, W * D, err) &&
                      dalloc(d, d.ext, ((size_t) g.win + W) * g.hd, err) && dalloc(d, d.xe, P * D, err) &&
                      dalloc(d, d.xg, P * g.ff, err) && dalloc(d, d.xu, P * g.ff, err) && dalloc(d, d.xhh, P * g.ff, err) &&
                      dalloc(d, d.ed, P * D, err) && dalloc(d, d.plan_ptr, P, err);
            if (!ok) return false;
            d.attn_scr.max_t = kAttnScratchT;
            d.attn_scr.cap = (int) topk_stride();
            if (!dalloc(d, d.attn_scr.rows, (size_t) kAttnScratchT * topk_stride() * g.hd, err)) return false;
            if (!g.eg_layers.empty()) {
                const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
                ok = dalloc(d, d.eg_rows, W * cols * (pk.eg_hd + pk.eg_hd / 32), err) &&
                     dalloc(d, d.eg_x, W * cols * pk.eg_hd, err) && dalloc(d, d.eg_kv, W * (g.hc + 1) * D, err);
                if (!ok) return false;
            }
            if ((int) di == head_dev && !dalloc(d, d.logits, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(d.bounce, bounce_bytes(), err)) return false;
            if (!dalloc(d, d.xcnt, 4, err) || !hmap(d.hand, bounce_bytes(), err) || !hmap(d.hand_flag, 32, err) ||
                !dalloc(d, d.grp_ptr, P, err) || !dalloc(d, d.grp_start, P + 1, err) || !dalloc(d, d.grp_ent, P, err) ||
                !dalloc(d, d.n_groups, 4, err))
                return false;
            if (g.n_mtp > 0 && !dalloc(d, d.tap, (size_t) tap_cap * D, err)) return false;
            if (g.n_mtp > 0 && (int) di == head_dev &&
                (!dalloc(d, d_confv, kWin, err) || !dalloc(d, d_ints, 64, err) || !dalloc(d, d_specq, kWin, err)))
                return false;
            CK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
            cudaSetDevice(d.id);
            CK(cudaMemcpy(d.freqs_win, freqs_win.data(), freqs_win.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d.freqs_comp, freqs_comp.data(), freqs_comp.size() * 4, cudaMemcpyHostToDevice));
            for (Src& s : srcs) {
                if (s.h_ckv == nullptr) {   // (its events belong to the source layer's device: it records them)
                    if (!halloc(s.h_ckv, (size_t) kWin * ckv_rb(), err) || !halloc(s.h_ik, (size_t) kWin * ik_rb(), err))
                        return false;
                    CK(cudaSetDevice(devs[(size_t) L[(size_t) s.layer].dev].id));
                    CK(cudaEventCreateWithFlags(&s.ev_ckv, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&s.ev_ik, cudaEventDisableTiming));
                    CK(cudaSetDevice(d.id));
                }
                if (s.ckv[di] != nullptr && !dalloc(d, s.ckv[di], (size_t) s.cap * ckv_rb(), err, &d.caches))
                    return false;
                if (s.ik[di] != nullptr && !dalloc(d, s.ik[di], (size_t) s.cap * ik_rb(), err, &d.caches))
                    return false;
                // the row-split prefill's own copy of every source on every device (the wavefront's per-chunk rows);
                // it stands in for ckv/ik only while a row-split prompt runs, so the pipeline keeps its own
                if (rp_do() && (!dalloc(d, s.ckv_rp[di], (size_t) s.cap * ckv_rb(), err, &d.caches) ||
                                !dalloc(d, s.ik_rp[di], (size_t) s.cap * ik_rb(), err, &d.caches)))
                    return false;
            }
        }
        for (Src& sr : srcs) {
            if (sr.remote && (!hmap(sr.x_stage, x_rows(sr) * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag, 32, err)))
                return false;
            if (sr.remote && chunk > 0 &&
                (!hmap(sr.x_stage1, x_rows(sr) * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag1, 32, err) ||
                 !hmap(sr.x_ack, (size_t) 2 * devs.size() * 32, err)))
                return false;
            const int sg = L[(size_t) sr.layer].grp;
            for (size_t di = 0; di < devs.size(); ++di)
                if ((sr.ckv[di] != nullptr || sr.ik[di] != nullptr) && devs[di].grp != sg) sr.readers.push_back((int) di);
        }
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            Group& G = groups[gi];
            G.ar.nm = (int) G.mem.size();
            G.ar.slots = 4;
            G.ar.stride = (size_t) kWin * g.dim;
            if (G.ar.nm > 1 && (!hmap(G.ar.data, (size_t) G.ar.nm * G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.flags, (size_t) G.ar.nm * G.ar.slots * 32, err)))
                return false;
            // three members or more: the reduce-scatter's sums and its per-block flags (allreduce)
            if (G.ar.nm > 2 && (!hmap(G.ar.sums, (size_t) G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.bflags, (size_t) 2 * G.ar.nm * G.ar.slots * kArBlocks * 8, err)))
                return false;
            if (G.ar.nm > 1 && &G == &groups[(size_t) devs[(size_t) head_dev].grp] &&
                (!hmap(G.lg_stage, (size_t) 2 * (G.ar.nm - 1) * kWin * ((g.vocab / 128 + G.ar.nm - 1) / G.ar.nm * 128), err) ||
                 !hmap(G.lg_flag, (size_t) 2 * (G.ar.nm - 1) * 32, err) || !hmap(G.lg_ack, (size_t) 2 * 32, err)))
                return false;
            G.arb = G.ar;   // (a chunk's: two slots of its group's step rows)
            G.arb.slots = 2;
            G.arb.stride = (size_t) chunk_steps[gi] * g.dim;
            if (chunk > 0 && G.arb.nm > 1 && (!hmap(G.arb.data, (size_t) G.arb.nm * G.arb.slots * G.arb.stride, err) ||
                                              !hmap(G.arb.flags, (size_t) G.arb.nm * G.arb.slots * 32, err)))
                return false;
            if (chunk > 0 && G.arb.nm > 2 &&
                (!hmap(G.arb.sums, (size_t) G.arb.slots * G.arb.stride, err) ||
                 !hmap(G.arb.bflags, (size_t) 2 * G.arb.nm * G.arb.slots * kArBlocks * 8, err)))
                return false;
        }
        if (g.n_mtp > 0) {
            if (!halloc(h_taps, (size_t) tap_cap * g.d_targets.size() * g.dim, err)) return false;
            CK(cudaSetDevice(devs[(size_t) d_dev].id));
            CK(cudaEventCreateWithFlags(&ev_seed, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_seed, devs[(size_t) d_dev].st));
        }
        // the dense weights, each device's layers by a thread of its own
        {
            std::vector<std::string> errs(devs.size());
            std::vector<std::thread> th;
            for (size_t di = 0; di < devs.size(); ++di)
                th.emplace_back([this, di, &errs] {
                    std::string e;
                    if (!load_dense((int) di, e)) errs[di] = e;
                });
            for (auto& t : th) t.join();
            for (auto& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        if (o.verbose)
            std::fprintf(stderr, "ds4: dense weights loaded in %.1f s\n", (now_ms() - t0) / 1000.0);
        // the markov steps over a head group of two or more (DS4_MK_SPLIT=0: the lead's whole rows, as before): E
        // host-mapped, the steps' slots
        // (only where every member holds its rows of the markov head: a type load_dense does not take keeps the lead's)
        bool mk_rows = g.n_mtp > 0 && head_nm() > 1;
        for (int m : groups[(size_t) devs[(size_t) head_dev].grp].mem) mk_rows = mk_rows && devs[(size_t) m].mk_share != nullptr;
        if (mk_rows && !(std::getenv("DS4_MK_SPLIT") != nullptr && std::atoi(std::getenv("DS4_MK_SPLIT")) == 0)) {
            const Tensor* te = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.embed.weight");
            if (te == nullptr || te->dtype != "BF16" || te->bytes != (uint64_t) g.vocab * g.d_rank * 2) {
                err = "the markov embed: not a BF16 [vocab][rank] table";
                return false;
            }
            if (!hmap(h_mk_embed, (size_t) g.vocab * g.d_rank, err) || !pk.read(*te, h_mk_embed) ||
                !hmap(mk_slots, (size_t) kWin * head_nm() * 4, err)) {
                if (err.empty()) err = "the markov embed: read failed";
                return false;
            }
            devs[(size_t) head_dev].mk_ints = d_ints;   // (the lead's drafted tokens: where the confidence head reads them)
            mk_split = true;
        }

        // the embedding table, pinned on the host
        {
            const Tensor* t = pk.get("embed.weight");
            const size_t nv = (size_t) g.vocab * g.dim;
            const bool q8 = t != nullptr && t->dtype == "Q8_0" && t->bytes == nv / 32 * gg::kQ8Bytes;
            if (t == nullptr || (!q8 && (t->dtype != "BF16" || t->bytes != nv * 2))) {
                err = "embed.weight: not a BF16 (or Q8_0) [vocab][dim] table";
                return false;
            }
            if (!halloc(embed_pinned, nv, err)) return false;
            if (q8) {   // (a GGUF's: each value rounded to BF16, a row at a time)
                std::vector<uint8_t> row((size_t) g.dim / 32 * gg::kQ8Bytes);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * row.size(), row.size(), row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    const std::vector<float> v = q8_values(row.data(), (size_t) g.dim);
                    for (size_t c = 0; c < (size_t) g.dim; ++c) {
                        uint32_t u;
                        std::memcpy(&u, &v[c], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
                        embed_pinned[r * g.dim + c] = (uint16_t) (u >> 16);
                    }
                }
            } else if (!pk.read(*t, embed_pinned)) {
                err = "embed.weight: read failed";
                return false;
            }
        }
        // the host side of the CPU lane and the Engram rows
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
        if (!halloc(h_x, (size_t) kWin * g.dim, err) || !halloc(h_w, (size_t) kWin * kPicks, err) ||
            !halloc(h_part, (size_t) kWin * g.dim, err) || !halloc(h_idx, (size_t) kWin * kPicks, err) ||
            !halloc(h_emb, (size_t) std::max(kWin, chunk) * g.dim, err) ||
            !halloc(h_eg, (size_t) std::max(1, pk.eg_layers) * eg_cap * cols * (pk.eg_hd + pk.eg_hd / 32) + 64, err))
            return false;
        if (pk.eg_layers > 0) {
            // the Engram rows' reader: threads enough to keep an SSD's queue busy (reads from the page cache need few);
            // the row cache a small share of the RAM free (DS4_ENGRAM_CACHE_MB: its size, 0 off; DS4_ENGRAM_THREADS)
            const size_t row = (size_t) pk.eg_hd + pk.eg_hd / 32;
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            int nth = (int) std::min(16u, std::max(4u, hw / 2));
            if (const char* e = std::getenv("DS4_ENGRAM_THREADS")) nth = std::max(1, std::atoi(e));
            const uint64_t avail = ram_avail();
            uint64_t mb = avail == ~uint64_t{0} ? 64 : std::min<uint64_t>(256, std::max<uint64_t>(16, (avail >> 20) / 64));
            if (const char* e = std::getenv("DS4_ENGRAM_CACHE_MB")) mb = std::strtoull(e, nullptr, 10);
            eg_cache.init((size_t) ((mb << 20) / row), row);
            std::vector<std::string> paths;
            for (const std::string& fn : pk.files) paths.push_back(pk.model_dir + "/" + fn);
            eg_io.start(paths, nth);
            for (int li = 0; li < pk.eg_layers; ++li) {
                eg_win.emplace_back(new EgBatch());
                const Tensor* tw = pk.get("layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.weight");
                const bool q6k = tw != nullptr && tw->dtype == "Q6_K";
                if (q6k && pk.eg_hd % 256 != 0) {
                    err = "Engram layer " + std::to_string(g.eg_layers[(size_t) li]) + ": Q6_K rows of " +
                          std::to_string(pk.eg_hd) + " values (not whole superblocks of 256)";
                    return false;
                }
                eg_q6k.push_back(q6k ? 1 : 0);
            }
            if (o.verbose)
                std::fprintf(stderr, "ds4: Engram rows read on %d threads, issued with the tokens, collected before "
                                     "their layer; row cache %llu MB (%zu rows)\n",
                             nth, (unsigned long long) mb, eg_cache.capacity());
        }
        prof.assign((size_t) (g.nl + g.n_mtp) * g.n_exp, 0u);
        if (!o.profile.empty()) {
            std::FILE* f = std::fopen(o.profile.c_str(), "rb");
            if (f != nullptr) {
                std::vector<uint32_t> p(prof.size());
                const size_t n = std::fread(p.data(), 4, p.size(), f);
                if (n == p.size() || n == (size_t) g.nl * g.n_exp) std::copy(p.begin(), p.begin() + (long) n, prof.begin());
                std::fclose(f);
            }
        }

        // the chunked prompt path's buffers (before the experts: they take what is left)
        if (chunk > 0 && !alloc_chunks(err)) return false;
        // the CPU lane's experts read in place (decided above): how much of them the load reads ahead, from what VRAM
        // now leaves over
        if (file_experts) {
            double room = 0, all = 0;   // the VRAM place_experts will find for experts; every expert's bytes
            for (size_t i = 0; i < devs.size(); ++i) {
                Dev& d = devs[i];
                cudaSetDevice(d.id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                room += std::max(0.0, (double) fr - (double) resv[i]) * o.vram_experts;
            }
            for (const Layer& l : L) all += (double) l.blob * l.n_exp;
            const double need = std::max(0.0, all - room);
            const uint64_t avail = ram_avail();
            warm_bytes = avail > kRamMargin ? std::min<uint64_t>(avail - kRamMargin, (uint64_t) need) : 0;
            if (avail == ~uint64_t{0}) warm_bytes = (uint64_t) need;
            std::fprintf(stderr,
                         "ds4: the experts VRAM leaves over (~%.1f GB) are read in place from the model's files%s "
                         "(%.1f GB of RAM free to keep them: the most-routed %.1f GB read ahead; prompts in chunks of %d)\n",
                         need / 1e9, o.mmap_experts > 0 ? " (--mmap-experts on)"
                                     : pk.gguf ? " (--mmap-experts off copies them into RAM)" : ", not copied into RAM",
                         avail == ~uint64_t{0} ? 0.0 : avail / 1e9, warm_bytes / 1e9, chunk);
        }
        // the CPU lane: one pool of workers per NUMA node, each expert split in NH column parts (part X on node X); in
        // place, one part (the files' layout) and one pool over every node's cores (the pages lie where the cache put them)
        cores = node_cores(&all_cpus);
        NH = file_experts ? 1 : (int) std::min<size_t>(2, cores.size());
        if (NH > 1 && ((g.ff / NH) % 128 != 0 || (g.dim / NH) % 128 != 0)) NH = 1;
        if (!pk.gguf) make_pools();   // (a GGUF's after its experts: copy_experts decides NH)
        {
            const size_t E = (size_t) g.dim, FF = (size_t) g.ff, GQ = E / 256, DQ = FF / 128, S = kSlots;
            c_xh.resize((size_t) NH * S * 2 * E);
            c_xsb.resize((size_t) NH * S * 2 * (E / 128));
            c_acc.resize((size_t) NH * S * 2 * GQ * (FF / NH));
            c_xhd.resize(S * FF);
            c_xsd.resize(S * (FF / 128));
            c_accd.resize((size_t) NH * S * DQ * (E / NH));
            c_yd.resize(S * E);
        }
        // the row-split prefill's replicas take their VRAM before the expert slots do (the mode gives up slots for
        // the layer groups: see rp_plan)
        if (chunk > 0 && !rp_plan(err)) return false;
        // the experts: VRAM slots in what each device has left, the rest on the host
        if (!place_experts(err)) return false;
        if (!load_experts(err)) return false;
        if (pk.gguf) make_pools();
        if (chunk > 0) {
            // the CPU lane's parts pinned (the chunks stream them to the GPUs by DMA), each layer's virtual slots
            const double tr = now_ms();
            for (auto& m : host_maps)
                if (cudaHostRegister(m.first, m.second, cudaHostRegisterPortable) != cudaSuccess) {
                    err = "pinning the CPU lane's experts (" + std::to_string(m.second >> 20) + " MiB) failed";
                    return false;
                }
            if (o.verbose) std::fprintf(stderr, "ds4: CPU lane's experts pinned in %.1f s\n", (now_ms() - tr) / 1000.0);
            if (file_experts) {   // in place: each device's staging for its ring, and its readers (DS4_READERS, 8)
                size_t mr = 0;
                for (const Layer& l : L)
                    if (!l.mtp) {
                        for (uint64_t r : l.region_bytes) mr = std::max<size_t>(mr, r);
                        if (l.lay.fmt == gg::kFmtMXFP4) {   // (a GGUF's: read as its blob)
                            mr = std::max(mr, l.blob);
                            for (int r = 0; r < 3; ++r) flow_part[r] = std::max(flow_part[r], mx_part_bytes(l, r));
                        }
                    }
                const char* nr = std::getenv("DS4_READERS");
                const int n_readers = std::max(1, nr != nullptr ? std::atoi(nr) : 8);
                const bool flow_on = std::getenv("DS4_FLOW") == nullptr || std::atoi(std::getenv("DS4_FLOW")) != 0;
                // (a tensor-parallel group's members on the flow only with DS4_FLOW_TP=1 for now: each reads the
                // whole down of every expert it streams, 25 MB an expert for a pair, 18.8 alone - 3x V100, the
                // 17-token prompt from the page cache: 2.5-2.7 s on the flow, 1.3-1.6 s in the group's windows)
                const bool flow_tp = std::getenv("DS4_FLOW_TP") != nullptr && std::atoi(std::getenv("DS4_FLOW_TP")) != 0;
                for (size_t di = 0; di < chunks.size(); ++di) {
                    auto& cp = chunks[di];
                    if (cp == nullptr || cp->ring == nullptr) continue;
                    cp->stage_stride = (mr + 4095) & ~(size_t) 4095;
                    // the flow: every trunk layer of the device a GGUF's (alone or a tensor-parallel group's member)
                    bool fl = flow_on;
                    int n_exp_max = 0;
                    for (const Layer& l : L)
                        if (!l.mtp)
                            for (int m = 0; m < l.nm; ++m)
                                if (member(l, m).dev == (int) di) {
                                    // (a group's members: each its share of the down's blocks in every row)
                                    fl = fl && l.lay.fmt == gg::kFmtMXFP4 &&
                                         (l.nm == 1 || (flow_tp && (g.ff / l.nm) %
                                                        std::max(1, gg::type_block(gg::expert_type(l.lay, 2))) == 0));
                                    n_exp_max = std::max(n_exp_max, l.n_exp);
                                }
                    if (fl)   // (each part read unbuffered from its offset rounded down: a page more each)
                        cp->stage_stride = std::max(cp->stage_stride, flow_stride());
                    cp->S = cp->ring_n;
                    if (!halloc(cp->stage, (size_t) cp->S * cp->stage_stride, err)) return false;
                    cp->readers = std::make_unique<Pool>(n_readers - 1, std::vector<int>{}, 50);
                    if (!fl) continue;
                    cudaSetDevice(devs[di].id);
                    cp->flow = true;
                    cp->dev = (int) di;
                    cp->io = std::make_unique<IoQueue>(n_readers);
                    cp->stage_item.assign((size_t) cp->S, -1);
                    cp->stage_ev.resize((size_t) cp->S);
                    cp->ring_fill.resize((size_t) cp->ring_n);
                    cp->ring_free.resize((size_t) cp->ring_n);
                    for (auto* v : {&cp->stage_ev, &cp->ring_fill, &cp->ring_free})
                        for (cudaEvent_t& e : *v) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                    // (a region is written again only bc_n batches later - more than a layer's items, so after that
                    // layer's bounds wait, which every copy from it precedes)
                    cp->bc_n = n_exp_max + 2;
                    cp->bc_w = cp->ring_n + 1;
                    if (!halloc(cp->bc_pin, (size_t) cp->bc_n * cp->bc_w, err) ||
                        !dalloc(devs[di], cp->bc_dev, (size_t) cp->bc_n * cp->bc_w, err))
                        return false;
                }
                // every device with layers on the flow: a prompt reads only the experts it picks, once, so a short one
                // takes the chunked path too (one pass, where windows of 16 read the CPU lane's picks again for each
                // window, a page fault at a time) - DS4_CHUNK_MIN still overrides
                bool all_flow = true;
                for (const auto& cp : chunks) all_flow = all_flow && (cp == nullptr || cp->ring == nullptr || cp->flow);
                if (all_flow && std::getenv("DS4_CHUNK_MIN") == nullptr) chunk_min = kFlowChunkMin;
            }
            for (Layer& l : L) {
                if (l.mtp) continue;
                for (int m = 0; m < l.nm; ++m) {   // (each member its own: the flow's members decide them each)
                    Layer& x = member(l, m);
                    if (!halloc(x.vs_pin, (size_t) l.n_exp, err)) return false;
                    if (!dalloc(devs[(size_t) x.dev], x.vs_dev, (size_t) l.n_exp, err)) return false;
                }
            }
            if (ep_on()) {   // expert-parallel prefill: every (layer, device) share's maps, and the host-mapped
                             // staging (one x | ridx | rw publication, one partial acc a device)
                int Cmax = 0;
                for (const auto& cp : chunks)
                    if (cp != nullptr) Cmax = std::max(Cmax, cp->C);
                ep_sh.resize(L.size());
                ep_x.resize(devs.size());
                ep_ridx.resize(devs.size());
                ep_rw.resize(devs.size());
                ep_x_flag.resize(devs.size());
                for (size_t di = 0; di < devs.size(); ++di)
                    if (!hmap(ep_x[di], (size_t) Cmax * g.dim, err) ||
                        !hmap(ep_ridx[di], (size_t) Cmax * kPicks, err) ||
                        !hmap(ep_rw[di], (size_t) Cmax * kPicks, err) || !hmap(ep_x_flag[di], 32, err))
                        return false;
                for (auto& cp : chunks) {
                    if (cp == nullptr) continue;
                    if (!hmap(cp->ep_acc, (size_t) cp->C * g.dim, err) || !hmap(cp->ep_acc_flag, 32, err))
                        return false;
                }
                for (size_t il = 0; il < L.size(); ++il) {
                    Layer& l = L[il];
                    if (l.mtp || l.vs_pin == nullptr) continue;
                    ep_sh[il].resize(devs.size());
                    for (size_t di = 0; di < devs.size(); ++di) {
                        EpShare& sh = ep_sh[il][di];
                        sh.di = (int) di;
                        if (!halloc(sh.vs_pin, (size_t) l.n_exp, err)) return false;
                        if (!dalloc(devs[di], sh.vs_dev, (size_t) l.n_exp, err)) return false;
                    }
                }
                ep_alloc = true;
                if (o.verbose)
                    std::fprintf(stderr, "ds4 ep: shares of %zu layers over %zu devices, %.2f GB of host-mapped staging\n",
                                 ep_sh.size(), devs.size(),
                                 (double) Cmax * g.dim * 4 * 2.0 * (double) devs.size() / 1e9);
            }
        }
        {
            void* m = nullptr;
            if (cudaHostAlloc(&m, sizeof(Req) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's requests";
                return false;
            }
            host_allocs.push_back(m);
            reqs = (Req*) m;
            if (cudaHostAlloc(&m, sizeof(Resp) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's answers";
                return false;
            }
            host_allocs.push_back(m);
            resps = (Resp*) m;
            std::memset((void*) reqs, 0, sizeof(Req) * kRing);
            std::memset((void*) resps, 0, sizeof(Resp) * kRing);
            for (auto& x : layer_of) x.store(-1);
            ahead_fl.assign(L.size() * 1024, 0);
            ahead_rank.assign(L.size() * 1024, 0);
            ahead_for.assign(L.size(), 0);
            const int core = !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
            service = std::thread([this, core] {
                if (core >= 0) pin_self(core);
                serve_lane();
            });
            if (ahead_mode > 0) pf_th = std::thread([this] { prefetch_loop(); });
        }
        if (o.expert_cache) {
            size_t maxb = 0;
            for (const Layer& l : L) maxb = std::max(maxb, l.blob);
            const int nt = std::max(1, std::min(c_inflight, std::getenv("DS4_CACHE_THREADS") ? std::atoi(std::getenv("DS4_CACHE_THREADS")) : 1));
            sw_bufs.resize((size_t) nt);
            for (SwapBufs& B : sw_bufs) {
                if (!halloc(B.a, maxb, err) || !halloc(B.b, maxb, err) || !halloc(B.h, maxb, err)) return false;
                for (Dev& d : devs) {
                    cudaSetDevice(d.id);
                    cudaStream_t cs = nullptr;
                    CK(cudaStreamCreateWithFlags(&cs, std::getenv("DS4_RP_CHECK") ? cudaStreamDefault : cudaStreamNonBlocking));
                    B.st.push_back(cs);
                }
            }
            for (SwapBufs& B : sw_bufs) cache_ths.emplace_back([this, &B] { cache_loop(B); });
            for (Dev& d : devs) {   // (the lend refill's own streams, apart from the cache's swaps)
                cudaSetDevice(d.id);
                cudaStream_t cs = nullptr;
                CK(cudaStreamCreateWithFlags(&cs, std::getenv("DS4_RP_CHECK") ? cudaStreamDefault : cudaStreamNonBlocking));
                copy_st.push_back(cs);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
        }
        if (par || chunk > 0)   // (the windows' threads; the prompt path's, whatever mode the windows take)
            for (size_t di = 0; di < devs.size(); ++di) {
                workers.push_back(std::make_unique<Worker>());
                workers.back()->start();
            }
        calibrate_paths();
        if (o.verbose) std::fprintf(stderr, "ds4: loaded in %.1f s\n%s", (now_ms() - t0) / 1000.0, describe().c_str());
        reset();
        return true;
    }


    // the caches' row bytes, packed as DeepSeek keeps them (ds4_kernels.cuh): the window ring's (FP8), the compressed
    // KV's (FP4 + E4M3 scales), the index keys' (FP4 + power-of-two scales)
    size_t win_rb() const { return kv_row_bytes(kKvFp8, g.hd); }
    size_t ckv_rb() const { return kv_row_bytes(kKvFp4, g.hd); }
    size_t ik_rb() const { return kv_row_bytes(kKvFp4p, g.ix_hd); }

    // ---- the chunked prompt path
    // the staging rows one chunk of the source's group needs (a consumer's chunk is whole chunks of the source's run)
    size_t x_rows(const Src& sr) const {
        const size_t gi = (size_t) L[(size_t) sr.layer].grp;
        return (size_t) std::max(kWin, (gi < chunk_steps.size() ? chunk_steps[gi] : chunk) + 2);
    }
    size_t chunk_hand_bytes(int C) const {
        return (size_t) C * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    // the VRAM a member of group gi's chunk buffers take (alloc_chunks' allocations, less the experts): the split
    // search keeps this free per member.  The per-token sum mirrors alloc_chunks' dalloc sizes; w16, the A-sized GEMM
    // slabs and the score rings are the fixed part; the ring is the streamed experts' read-ahead slots
    size_t chunk_vram_est(int gi) const {
        const int C = gi < (int) chunk_steps.size() ? chunk_steps[(size_t) gi] : chunk;
        if (C <= 0) return 0;
        const int nm = (int) groups[(size_t) gi].mem.size();
        const int HQ = g.heads / nm * g.hd, OG = g.o_groups / nm, FF = g.ff / nm;
        const size_t A = (size_t) std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
        const size_t AI = (size_t) std::max(32, std::min(128, (int) (((size_t) 64 << 20) /
                                    (((size_t) o.max_context + 64) * 4)) / 32 * 32));
        const size_t Cc = (size_t) C, D = (size_t) g.dim, HD = (size_t) g.hd, H = (size_t) g.hc;
        const size_t ql = (size_t) g.q_lora;
        const size_t per_tok = Cc * (H * D * 4 + D * 4 * 3 + D * 2 + ql * 4 + ql * 2 + HD * 4 * 5 + HD * 2 * 2 +
                                     (size_t) g.ix_hd * 4 + 64 * 4 + (size_t) topk_stride() * 4 +
                                     (size_t) cand_stride() * 4 + 32 * 4 + 8 * 4 * 4 + 64 * 4 * 2 + 8 * 4 + 1024 * 4 +
                                     (size_t) kPicks * 4 * 4 + (size_t) FF * 4 * 2 + (size_t) FF * 2 * 2);
        const size_t fixed = ((size_t) 8 << 20) * 4 + A * (size_t) HQ * 4 * 3 +
                             A * (size_t) OG * (size_t) g.o_lora * 4 * 2 +
                             AI * (size_t) g.ix_heads * g.ix_hd * 4 + AI * ((size_t) o.max_context + 64) * 4 +
                             AI * bm_stride() * 4 + 2048 * 4 + (size_t) g.win * HD * 4;
        size_t stride = 0;
        for (const Layer& l : L)
            if (!l.mtp) stride = std::max(stride, align256(unit(l)));
        size_t ring = 0;
        if (stride > 0) {
            const size_t mb = std::getenv("DS4_RING_MB") ? (size_t) std::atoi(std::getenv("DS4_RING_MB")) : 400;
            ring = (size_t) std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts) * stride;
        }
        const size_t eg = pk.eg_layers > 0
                              ? Cc * (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32)
                              : 0;
        return per_tok + fixed + ring + (gi == 0 ? Cc * D * 2 : 0) + eg;
    }

    // the chunk path's device buffers, in one list: alloc_chunks runs it twice - once with no arena yet (a sizing
    // pass, whose sum is the arena's exact bytes), once to carve them out of the arena (carve: zeroed, 256-byte
    // aligned - the same addresses the old per-buffer cudaMallocs would have taken)
    bool chunk_gpu_bufs(Dev& d, Chunk& c, bool has_first, bool has_eg, size_t stride, std::string& err) {
        const int D = g.dim, H = g.hc, HD = g.hd;
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        const int nm = (int) groups[(size_t) d.grp].mem.size();
        const int NHL = g.heads / nm, HQ = NHL * HD, OG = g.o_groups / nm, FF = g.ff / nm;
        const size_t C = (size_t) c.C, A = (size_t) c.A, AI = (size_t) c.AI, ss = (size_t) o.max_context + 64;
        if (d.arena != nullptr) d.arena_spans.clear();   // (the carving pass records it; the sizing pass has no arena)
        // row-major buffers take the prompt's T rows at their start: 0 + per-token stride * T (the context-wide ones -
        // scores, bm - and the fixed-batch ones - q, o, iq - stay whole: nothing is read past what is zeroed)
        bool ok = carve(d, c.h, C * H * D, err, 0, (size_t) H * D) && carve(d, c.x, C * D, err, 0, (size_t) D) &&
                  carve(d, c.acc, C * D, err, 0, (size_t) D) && carve(d, c.x16, C * D, err, 0, (size_t) D) &&
                  carve(d, c.qr, C * g.q_lora, err, 0, (size_t) g.q_lora) &&
                  carve(d, c.qr16, C * g.q_lora, err, 0, (size_t) g.q_lora) &&
                  carve(d, c.kvr, C * HD, err, 0, (size_t) HD) && carve(d, c.ckr, C * HD, err, 0, (size_t) HD) &&
                  carve(d, c.cgr, C * HD, err, 0, (size_t) HD) &&
                  carve(d, c.latent, (C + 2) * HD, err, 2 * (size_t) HD, (size_t) HD) &&
                  carve(d, c.lat16, (C + 2) * HD, err, 2 * (size_t) HD, (size_t) HD) &&
                  carve(d, c.ikr, (C + 2) * g.ix_hd, err, 2 * (size_t) g.ix_hd, (size_t) g.ix_hd) &&
                  carve(d, c.ext, ((size_t) g.win + C) * HD, err, (size_t) g.win * HD, (size_t) HD) &&
                  carve(d, c.q, A * HQ, err) && carve(d, c.o, A * HQ, err) && carve(d, c.o16, A * HQ, err) &&
                  carve(d, c.oa, A * OG * g.o_lora, err) && carve(d, c.oa16, A * OG * g.o_lora, err) &&
                  carve(d, c.iq, AI * g.ix_heads * g.ix_hd, err) && carve(d, c.iw, C * 64, err, 0, 64) &&
                  carve(d, c.scores, AI * ss, err) && carve(d, c.bm, AI * bm_stride(), err) &&
                  carve(d, c.topk, C * topk_stride(), err, 0, (size_t) topk_stride()) &&
                  carve(d, c.cand, C * cand_stride(), err, 0, (size_t) cand_stride()) &&
                  carve(d, c.mix, C * 32, err, 0, 32) && carve(d, c.a_pre, C * 8, err, 0, 8) &&
                  carve(d, c.a_post, C * 8, err, 0, 8) && carve(d, c.a_comb, C * 64, err, 0, 64) &&
                  carve(d, c.f_pre, C * 8, err, 0, 8) && carve(d, c.f_post, C * 8, err, 0, 8) &&
                  carve(d, c.f_comb, C * 64, err, 0, 64) && carve(d, c.pre, C * 8, err, 0, 8) &&
                  carve(d, c.glog, C * 1024, err, 0, 1024) &&
                  carve(d, c.ridx, C * kPicks, err, 0, (size_t) kPicks) &&
                  carve(d, c.rw, C * kPicks, err, 0, (size_t) kPicks) && carve(d, c.bounds, 2048, err) &&
                  carve(d, c.row_tok, C * kPicks, err, 0, (size_t) kPicks) &&
                  carve(d, c.pos, C * kPicks, err, 0, (size_t) kPicks) &&
                  carve(d, c.sg, C * FF, err, 0, (size_t) FF) && carve(d, c.su, C * FF, err, 0, (size_t) FF) &&
                  carve(d, c.sh16, C * FF, err, 0, (size_t) FF) && carve(d, c.h16, C * FF, err, 0, (size_t) FF) &&
                  carve(d, c.y, C * D, err, 0, (size_t) D) && carve(d, c.w16, c.w16_elems, err);
        if (!ok) return false;
        if (has_first && !carve(d, c.emb, C * D, err, 0, (size_t) D)) return false;
        if (has_eg) {
            const size_t ES = 256, per = (size_t) cols * (ehd + ehd / 32);
            if (!carve(d, c.eg_x, ES * cols * ehd, err) || !carve(d, c.eg_x16, ES * cols * ehd, err) ||
                !carve(d, c.eg_kv, ES * (H + 1) * D, err) || !carve(d, c.eg_rows, C * per, err, 0, per))
                return false;
        }
        if (stride > 0 && !carve(d, c.ring, (size_t) c.ring_n * stride, err)) return false;
        return true;
    }

    bool alloc_chunks(std::string& err) {
        chunks.resize(devs.size());
        const int D = g.dim, H = g.hc, HD = g.hd;
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            chunks[di] = std::make_unique<Chunk>();
            Chunk& c = *chunks[di];
            const int nm = (int) groups[(size_t) d.grp].mem.size();
            const int NHL = g.heads / head_members(nm), HQ = NHL * HD, OG = g.o_groups / head_members(nm);
            int FF = 0, ff0_ = 0;
            split128(g.ff, nm, 0, ff0_, FF);   // (the widest member's share: member 0 takes the odd blocks)
            const size_t C = (size_t) chunk_steps[(size_t) d.grp];
            c.C = (int) C;
            c.A = std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
            // (the indexer's slab: its scores rows are the context wide - about 64 MB of them)
            c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
            c.w16_elems = (size_t) 8 << 20;
            bool has_eg = false, has_first = false;
            size_t stride = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di && !l.mtp) {
                        has_eg = has_eg || l.eg >= 0;
                        has_first = has_first || l.il == 0;
                        stride = std::max(stride, align256(unit(member(l, m))));
                    }
            if (ep_on())   // (expert-parallel: a device streams other groups' layers too - the widest expert of any)
                for (const Layer& l : L)
                    if (!l.mtp)
                        for (int m = 0; m < l.nm; ++m) stride = std::max(stride, align256(unit(member(l, m))));
            if (rp_do()) {   // (row-split: every device runs every layer in some stage - layer 0's embedding, Engram,
                has_first = true;   // and the widest expert of any layer, not only of its own)
                has_eg = has_eg || !g.eg_layers.empty();
                for (const Layer& l : L)
                    if (!l.mtp)
                        for (int m = 0; m < l.nm; ++m) stride = std::max(stride, align256(unit(member(l, m))));
            }
            // --chunk-mmq: MMQ addresses an expert's blob in whole type blocks (MXFP4: 17 bytes), so the slot stride
            // has to be a multiple of that as well as 256 (4352 = 256 * 17); the experts keep their own bytes
            if (mmq_do() && stride > 0) stride = (stride + 4351) / 4352 * 4352;
            if (stride > 0) {   // DS4_RING_MB (400) of slots in kParts batches
                const size_t mb = std::getenv("DS4_RING_MB") ? (size_t) std::atoi(std::getenv("DS4_RING_MB")) : 400;
                c.ring_stride = stride;
                c.ring_n = std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts);
            }
            // the device buffers, from one arena of exactly their bytes: a sizing pass, one cudaMalloc, the carve.
            // Between prompts the same bytes are the experts' lend banks (Layer::lend_base).
            d.arena = nullptr;
            d.arena_bytes = d.arena_off = 0;
            if (!chunk_gpu_bufs(d, c, has_first, has_eg, stride, err)) return false;
            if (d.arena_bytes > 0) {
                void* v = nullptr;
                if (cudaMalloc(&v, d.arena_bytes) != cudaSuccess) {
                    err = "cudaMalloc " + std::to_string(d.arena_bytes >> 20) + " MiB (the prompt arena) on CUDA" +
                          std::to_string(d.id) + " failed";
                    return false;
                }
                d.arena = (uint8_t*) v;
                d.allocs.push_back(v);
                if (!chunk_gpu_bufs(d, c, has_first, has_eg, stride, err)) return false;
                if (d.arena_off != d.arena_bytes) {   // (the two passes must take the same list)
                    err = "internal: CUDA" + std::to_string(d.id) + "'s prompt arena was sized " +
                          std::to_string(d.arena_bytes) + " but carved " + std::to_string(d.arena_off) + " bytes";
                    return false;
                }
            }
            if (!halloc(c.bounds_pin, 2048, err)) return false;
            CK(cudaStreamCreateWithFlags(&c.cs, std::getenv("DS4_RP_CHECK") ? cudaStreamDefault : cudaStreamNonBlocking));
            for (int i = 0; i < Chunk::kParts; ++i) {
                CK(cudaEventCreateWithFlags(&c.filled[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.freed[i], cudaEventDisableTiming));
            }
            if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS) {
                err = "cuBLAS on CUDA" + std::to_string(d.id);
                return false;
            }
            for (int sl = 0; sl < 2; ++sl) {
                if (d.mi == 0 && (!hmap(c.hand[sl], chunk_hand_bytes((int) C), err) || !hmap(c.hand_flag[sl], 32, err)))
                    return false;
                if (has_first && !halloc(c.emb_host[sl], C * D, err)) return false;
                if (has_eg && !halloc(c.eg_host[sl], C * pk.eg_layers * (size_t) cols * (ehd + ehd / 32), err)) return false;
                CK(cudaEventCreateWithFlags(&c.emb_ev[sl], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.eg_ev[sl], cudaEventDisableTiming));
            }
            if (d.mi == 0 && !hmap(c.hand_ack, (size_t) 2 * 8 * 32, err)) return false;
        }
        return true;
    }

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, cuBLAS
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err) {
        if (rows <= 0) return true;
        const int cols = std::max(128, (int) std::min<size_t>((size_t) m.n, c.w16_elems / (size_t) m.k) / 128 * 128);
        for (int c0 = 0; c0 < m.n; c0 += cols) {
            const int nc = std::min(cols, m.n - c0);
            if (m.fmt != 0) gg::rows(nc == m.n ? m : gg::view(m, c0, nc), c.w16, false, d.st);   // (a GGUF's)
            else x3::reconstruct_rows(nc == m.n ? m : x3::view(m, c0, nc), c.w16, false, d.st);
            const float one = 1.0f;
            const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, c.w16, CUDA_R_16F,
                                                   m.k, x16, CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                err = "cublasGemmEx: status " + std::to_string((int) st);
                return false;
            }
        }
        return true;
    }
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err) {
        const float zero = 0.0f;
        const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k,
                                               x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
            return false;
        }
        return true;
    }
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err) {
        const int hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32) !=
            CUBLAS_STATUS_SUCCESS) {
            err = "cublasSgemm (mHC mixes)";
            return false;
        }
        hc_mix_scale(h, hcd, n_mix, T, g.eps, mixes, d.st);
        return true;
    }

    // expert e of l (on the CPU lane: its NH column parts) -> member m's VRAM copy at dst, in the GPU layout (the full
    // blob, or a group member's part), by DMA from the pinned parts on the copy stream (in place: from `src`, its file
    // region read into pinned staging)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr) {
        if (l.lay.fmt == gg::kFmtMXFP4) {   // the blob, or member m's rows / blocks: from `src` (the expert read into
            // pinned staging as its blob) or from the file's mapping
            auto mat_r = [&](int r) { return src != nullptr ? src + l.lay.trellis[r] : mx_mat(l, e, r); };
            bool ok = true;
            if (l.nm == 1) {
                if (src != nullptr)
                    ok = cudaMemcpyAsync(dst, src, l.blob, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                else
                    for (int r = 0; r < 3; ++r)
                        ok = ok && cudaMemcpyAsync(dst + l.lay.trellis[r], mat_r(r), mx_part_bytes(l, r),
                                                   cudaMemcpyHostToDevice, cs) == cudaSuccess;
            } else {
                const int ffh = g.ff / l.nm;
                const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                             dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                             dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
                for (int r = 0; r < 2; ++r)
                    ok = ok && cudaMemcpyAsync(dst + l.hlay.trellis[r], mat_r(r) + (size_t) m * ffh * gr,
                                               (size_t) ffh * gr, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                ok = ok && cudaMemcpy2DAsync(dst + l.hlay.trellis[2], dh, mat_r(2) + (size_t) m * dh, dr, dh,
                                             (size_t) g.dim, cudaMemcpyHostToDevice, cs) == cudaSuccess;
            }
            if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
            return ok;
        }
        const auto& parts = l.hh[(size_t) e];
        const int D = g.dim, FF = g.ff;
        auto cp2 = [&](uint8_t* d0, size_t dp, const uint8_t* s0, size_t sp, size_t w, size_t h) {
            return cudaMemcpy2DAsync(d0, dp, s0, sp, w, h, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        auto cp = [&](uint8_t* d0, const uint8_t* s0, size_t n) {
            return cudaMemcpyAsync(d0, s0, n, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        bool ok = true;
        if (NH == 1) {   // one part: the whole expert in exllamav3's layout, its pieces at hl's offsets
            const uint8_t* p = src != nullptr ? src : parts[0];
            if (l.nm == 1) {
                for (int r = 0; r < 3; ++r)
                    ok = ok && cp(dst + l.lay.trellis[r], p + l.hl.trellis[r], l.piece_bytes[r][0]) &&
                         cp(dst + l.lay.suh[r], p + l.hl.suh[r], (size_t) l.hl.k[r] * 2) &&
                         cp(dst + l.lay.svh[r], p + l.hl.svh[r], (size_t) l.hl.n[r] * 2);
            } else {   // member m's part (its FFN columns [f0, f0 + fn)), cut as blob_to_half cuts a blob
                const Layer& x = member(l, m);
                const int f0 = x.ff0, fn = x.ffn;
                for (int r = 0; r < 2; ++r) {
                    const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) FF / 16, nth = (size_t) fn / 16;
                    ok = ok && cp2(dst + x.hlay.trellis[r], nth * tile, p + l.hl.trellis[r] + (size_t) (f0 / 16) * tile,
                                   nt * tile, nth * tile, (size_t) D / 16) &&
                         cp(dst + x.hlay.suh[r], p + l.hl.suh[r], (size_t) D * 2) &&
                         cp(dst + x.hlay.svh[r], p + l.hl.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
                }
                const size_t row = (size_t) (D / 16) * 32 * l.lay.K[2];
                ok = ok && cp(dst + x.hlay.trellis[2], p + l.hl.trellis[2] + (size_t) (f0 / 16) * row,
                              (size_t) (fn / 16) * row) &&
                     cp(dst + x.hlay.suh[2], p + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2) &&
                     cp(dst + x.hlay.svh[2], p + l.hl.svh[2], (size_t) D * 2);
            }
        } else if (l.nm == 1) {   // the full blob: each matrix's columns from the two parts side by side
            for (int r = 0; r < 3; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int k = l.hl.k[r], nh = l.hl.n[r];   // part: k x nh
                for (int X = 0; X < 2; ++X) {
                    ok = ok && cp2(dst + l.lay.trellis[r] + (size_t) X * (nh / 16) * tile, (size_t) (2 * nh / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r], (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile,
                                   (size_t) (k / 16));
                    ok = ok && cp(dst + l.lay.svh[r] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[r], (size_t) nh * 2);
                }
                ok = ok && cp(dst + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            }
        } else {   // member m (FFN columns [f0, f0 + fn)): gate / up from the parts holding them; down rows [f0, ..) of
                   // both parts' columns
            const Layer& x = member(l, m);
            const int f0 = x.ff0, fn = x.ffn;
            for (int r = 0; r < 2; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int hn = l.hl.n[r];   // part X: columns [X hn, (X + 1) hn)
                for (int X = 0; X < 2; ++X) {
                    const int a = std::max(f0, X * hn), b = std::min(f0 + fn, (X + 1) * hn);
                    if (a >= b) continue;
                    ok = ok && cp2(dst + x.hlay.trellis[r] + (size_t) ((a - f0) / 16) * tile, (size_t) (fn / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r] + (size_t) ((a - X * hn) / 16) * tile,
                                   (size_t) (hn / 16) * tile, (size_t) ((b - a) / 16) * tile, (size_t) (D / 16)) &&
                         cp(dst + x.hlay.svh[r] + (size_t) (a - f0) * 2, parts[(size_t) X] + l.hl.svh[r] + (size_t) (a - X * hn) * 2,
                            (size_t) (b - a) * 2);
                }
                ok = ok && cp(dst + x.hlay.suh[r], parts[0] + l.hl.suh[r], (size_t) D * 2);
            }
            const size_t tile = (size_t) 32 * l.lay.K[2];
            const int nh = D / 2;
            for (int X = 0; X < 2; ++X) {
                ok = ok && cp2(dst + x.hlay.trellis[2] + (size_t) X * (nh / 16) * tile, (size_t) (D / 16) * tile,
                               parts[(size_t) X] + l.hl.trellis[2] + (size_t) (f0 / 16) * (nh / 16) * tile,
                               (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile, (size_t) (fn / 16));
                ok = ok && cp(dst + x.hlay.svh[2] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[2], (size_t) nh * 2);
            }
            ok = ok && cp(dst + x.hlay.suh[2], parts[0] + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
        return ok;
    }

    // a device's streamed experts: batches of ring_n / kParts of each of its layers' CPU-lane experts, chunk by chunk
    // in layer order; the copies run at most kParts batches ahead of the compute
    struct Batch {
        int k = 0, il = 0, j0 = 0, n = 0;   // chunk, layer, its list's range
        const std::vector<int>* list = nullptr;   // whose experts: the layer's CPU lane, or an expert-parallel share's
        const Layer* lay = nullptr;   // the row-split prefill's replica for il (null: the split's own Layer)
    };
    struct Streamer {
        std::vector<Batch> b;
        size_t cp = 0, kp = 0;        // the next batch to copy, to compute
        size_t cc = 0, kc = 0, out = 0;   // expert-parallel: the batches copied, computed, and in flight (skipped ones aside)
        std::atomic<size_t> done{0};  // cp, for read_ahead's thread
    };
    // in place (file_experts): the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
        (void) di;   // (the batches are this device's: its layers' list, or its expert-parallel shares')
        static const double cap = (std::getenv("DS4_READAHEAD_MB") ? std::atof(std::getenv("DS4_READAHEAD_MB")) : 2048.0) *
                                  1048576.0;
        std::deque<std::pair<size_t, double>> issued;   // (batch, bytes) not yet copied
        double ahead = 0;
        for (size_t q = 0; q < S.b.size() && !stop.load(std::memory_order_relaxed);) {
            const size_t done = S.done.load(std::memory_order_relaxed);
            while (!issued.empty() && issued.front().first < done) {
                ahead -= issued.front().second;
                issued.pop_front();
            }
            if (ahead > cap) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const Batch& bt = S.b[q];
            const Layer& l0 = bt.lay != nullptr ? *bt.lay : L[(size_t) bt.il];
            const std::vector<int>& lst = *bt.list;
            double nb = 0;
            for (int i = 0; i < bt.n; ++i) {
                const int e = lst[(size_t) (bt.j0 + i)];
                if (l0.lay.fmt == gg::kFmtMXFP4) {
                    if (copy_mode && l0.ram[(size_t) e] >= 0) continue;   // (in RAM: nothing to read)
                    for (int r = 0; r < 3; ++r)
                        os::advise(pk.fds[(size_t) l0.mx_file[r]], (l0.mx_off[r] + (uint64_t) e * l0.mx_stride[r]), mx_part_bytes(l0, r), true);
                    nb += (double) l0.blob;
                    continue;
                }
                os::advise(pk.fds[(size_t) l0.region_file[(size_t) e]], l0.region_off[(size_t) e], l0.region_bytes[(size_t) e], true);
                nb += (double) l0.region_bytes[(size_t) e];
            }
            issued.push_back({q++, nb});
            ahead += nb;
        }
    }
    // (expert-parallel: `cur` and its bounds pb (vb: its first host slot) say what of the share is read - a range no row
    //  of the prompt picks is passed over with no read at all, and a layer whose routing is not known yet is not read
    //  ahead into; elsewhere the plan is copied as it stands)
    bool copy_ahead(Dev& d, Chunk& c, Streamer& S, int di, std::string& err,
                    const Layer* cur = nullptr, const int* pb = nullptr, int vb = 0) {
        const int P = Chunk::kParts;
        while (S.cp < S.b.size()) {
            const Batch& bt = S.b[S.cp];
            if (ep_run) {
                if (cur == nullptr) break;
                if (bt.il != cur->il) {
                    if (bt.il < cur->il) {   // (a layer already done: its left-over ranges)
                        ++S.cp;
                        S.done.store(S.cp, std::memory_order_relaxed);
                        continue;
                    }
                    break;   // (the next layer's share waits for its own routing)
                }
                const int v0 = vb + bt.j0;
                if (pb[(size_t) v0] == pb[(size_t) v0 + bt.n]) {   // (no row picks an expert of this range)
                    ++S.cp;
                    S.done.store(S.cp, std::memory_order_relaxed);
                    continue;
                }
                if (S.out >= (size_t) P) break;   // (P picked ranges of the share in the ring already)
            } else if (S.cp >= S.kp + P) {
                break;
            }
            const int half = ep_run ? (int) (S.cc++ % P) : (int) (S.cp % P);
            const Layer& l0 = bt.lay != nullptr ? *bt.lay : L[(size_t) bt.il];
            int mi = 0;
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di) mi = m;
            if (ep_run ? S.cc > (size_t) P : S.cp >= (size_t) P) CK(cudaStreamWaitEvent(c.cs, c.freed[half], 0));
            uint8_t* sb = nullptr;   // in place: the batch's regions in the part's staging slots
            const std::vector<int>& lst = *bt.list;
            if (c.stage != nullptr) {
                if (ep_run ? S.cc > (size_t) P : S.cp >= (size_t) P)
                    CK(cudaEventSynchronize(c.filled[half]));   // (its last batch's copies out)
                sb = c.stage + (size_t) half * (c.ring_n / P) * c.stage_stride;
                std::atomic<bool> bad{false};
                c.readers->run(bt.n, [&](int i) {
                    const int e = lst[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4 ? !read_mx(l0, e, sb + (size_t) i * c.stage_stride)
                                                    : !pk.read_at(l0.region_file[(size_t) e], l0.region_off[(size_t) e],
                                                                  l0.region_bytes[(size_t) e],
                                                                  sb + (size_t) i * c.stage_stride))
                        bad.store(true);
                });
                if (bad.load()) {
                    err = "reading layer " + std::to_string(l0.il) + "'s experts from the model's files failed";
                    return false;
                }
            }
            for (int i = 0; i < bt.n; ++i) {
                uint8_t* dst = c.ring + ((size_t) half * (c.ring_n / P) + i) * c.ring_stride;
                if (!stream_expert(l0, mi, lst[(size_t) (bt.j0 + i)], dst, c.cs, err,
                                   sb != nullptr ? sb + (size_t) i * c.stage_stride : nullptr))
                    return false;
                c.stream_bytes += (double) unit(member(l0, mi));
            }
            CK(cudaEventRecord(c.filled[half], c.cs));
            ++S.cp;
            if (ep_run) ++S.out;
            S.done.store(S.cp, std::memory_order_relaxed);
        }
        return true;
    }

    // ---- the flow (see Chunk)
    // Windows' HIP runtime submits queued work only when the host waits on the GPU: the flow's copies and the compute
    // it queued between its waits are submitted at once (cudaStreamQuery does that without waiting); elsewhere nothing
    static void kick(cudaStream_t s) {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
        (void) cudaStreamQuery(s);
        (void) cudaGetLastError();
#else
        (void) s;
#endif
    }
    size_t flow_region(int r) const {   // a part's room in a staging slot (read from its offset rounded down)
        return ((flow_part[r] + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1)) + os::kDirectAlign;
    }
    size_t flow_stride() const { return flow_region(0) + flow_region(1) + flow_region(2); }
    static Chunk::Item& fit(Chunk& c, size_t i) { return c.items[i - c.base]; }
    // what member m of l's group reads of matrix r of an expert: gate / up its rows [m n_ff / nm, ..) - one run, from
    // this offset in the matrix, this long; the down the whole matrix (its part, the blocks [m n_ff / nm / 32, ..) of
    // every row, is DMA'd out of it row by row); alone (nm 1) every matrix whole
    size_t mx_read_off(const Layer& l, int m, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) m * (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : 0;
    }
    size_t mx_read_len(const Layer& l, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : mx_part_bytes(l, r);
    }
    // the device's member index in L0's group (0 alone)
    int flow_mi(const Chunk& c, const Layer& L0) const {
        for (int m = 0; m < L0.nm; ++m)
            if (member(L0, m).dev == c.dev) return m;
        return 0;
    }
    void flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v) {
        c.items.emplace_back();
        Chunk::Item& it = c.items.back();
        it.il = l.il;
        it.mi = mi;
        it.k = k;
        it.e = e;
        it.v = v;
        const bool in_ram = !l.ram.empty() && l.ram[(size_t) e] >= 0;
        if (in_ram && ram_pinned) {   // (DMA'd from where it lies)
            const uint8_t* p = ram_slots[(size_t) l.ram[(size_t) e]];
            for (int r = 0; r < 3; ++r) it.src[r] = p + l.lay.trellis[r] + mx_read_off(l, mi, r);
            ++c.it_ram;
            return;
        }
        if (in_ram)   // (copied into staging by io, then DMA'd; copy_mode: from its NH parts, flow_pump)
            it.ram = copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] : ram_slots[(size_t) l.ram[(size_t) e]];
        bool hit = false;
        if (!in_ram && c.lc != nullptr && l.nm == 1) {   // layer-major: from the layer cache, or kept for the chunks after
            const size_t i = c.base + c.items.size() - 1, stride = (l.blob + 63) & ~(size_t) 63;
            if (c.lc_il != l.il) {   // (a layer's first item: the slots the layer before used are its own again
                c.lc_il = l.il;      // once that layer's items are all read - `after` below)
                c.lc_used = 0;
                c.lc_n = (int) std::min<size_t>(c.lc_of.size(), c.lc_bytes / stride);
                c.lc_first = i;
                std::fill(c.lc_of.begin(), c.lc_of.end(), -1);
            }
            const int j = c.lc_of[(size_t) e];
            if (j >= 0) {   // (copied into staging by io once its reader is done, then DMA'd)
                it.ram = c.lc + (size_t) j * stride;
                it.after = c.lc_writer[(size_t) j];
                hit = true;
            } else if (k + 1 < (int) c.fl_T.size() && c.lc_used < c.lc_n) {   // (a later chunk may pick it)
                const int jn = c.lc_used++;
                c.lc_of[(size_t) e] = jn;
                c.lc_writer[(size_t) jn] = (int64_t) i;
                it.keep = c.lc + (size_t) jn * stride;
                it.after = (int64_t) c.lc_first - 1;
            }
        }
        it.stage = (int) (c.n_file++ % (uint64_t) c.S);
        uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = l.mx_off[r] + (uint64_t) e * l.mx_stride[r] + mx_read_off(l, mi, r);
            it.src[r] = sb + (in_ram || hit ? 0 : off % os::kDirectAlign);
            sb += flow_region(r);
        }
        it.left.store(in_ram && copy_mode ? 3 * NH : 3, std::memory_order_relaxed);
        ++(in_ram ? c.it_ram : hit ? c.it_lc : c.it_file);
    }
    // the likely experts of (chunk k, layer il) for T tokens, first in its stream: the CPU lane's experts whose chance
    // to be picked at least once, 1 - (1 - share)^(picks), is at least DS4_SPEC_Q (0.9: the reads are what takes the
    // time, an unpicked one costs a whole read, and the routing is known a few ms after the layer before is done -
    // early reads pay where a chunk's long attention keeps the disk otherwise idle, and there nearly every expert is
    // that likely), likeliest first, at most what the staging and the ring hold (no more can be read before the layer
    // routes); the layer's virtual slots put them first, then its other CPU-lane experts in index order (the order the
    // rest of its picks are streamed in)
    // (a tensor-parallel group's members each decide their own stream and virtual slots - the same, from the same
    // profile and the same routing - and each reads only its part of every expert)
    void flow_spec(Chunk& c, int k, int il, int T) {
        static const double spec_q = std::getenv("DS4_SPEC_Q") ? std::atof(std::getenv("DS4_SPEC_Q")) : 0.9;
        Layer& l = lof(c, il);
        const int mi = flow_mi(c, l);
        int* vs = member(l, mi).vs_pin;
        c.spec_k = k;
        c.spec_il = il;
        c.spec_begin = c.base + c.items.size();
        const uint32_t* pr = prof.data() + (size_t) il * g.n_exp;
        double sum = 0;
        for (int e = 0; e < l.n_exp; ++e) sum += pr[e];
        std::vector<std::pair<double, int>> cand;
        const double picks = (double) l.k_exp * T;
        // (early reads only while they hit: once a run's likely ones went unpicked more often than DS4_SPEC_WASTE
        // (0.2) of the time, the rest of the run reads only what the routing picked - an unpicked one costs a whole
        // read, and how well the profile guesses depends on the PC's history and the prompt: V100, 64 tokens - 376
        // of 921 likely ones unpicked, 7 GB of 84 read for nothing; Radeon 8065S, 128 and 512 tokens - 44 of 945 and
        // 21 of 1568, the early reads 9-12% faster there)
        static const double waste = std::getenv("DS4_SPEC_WASTE") ? std::atof(std::getenv("DS4_SPEC_WASTE")) : 0.2;
        const bool early = c.it_spec < 2u * (uint64_t) c.ring_n || (double) c.it_spec_unused <= waste * (double) c.it_spec;
        for (int e : l.host_list) {
            if (!early) break;
            const double p = sum > 0 ? pr[e] / sum : 0.0;
            const double q = p >= 1.0 ? 1.0 : 1.0 - std::exp(picks * std::log1p(-p));
            if (q >= spec_q && spec_q <= 1.0) cand.push_back({q, e});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
        });
        if (cand.size() > (size_t) (c.S + c.ring_n)) cand.resize((size_t) (c.S + c.ring_n));
        std::vector<char> is((size_t) l.n_exp, 0);
        int j = 0;
        for (const auto& x : cand) {
            vs[x.second] = l.n_slots + j++;
            is[(size_t) x.second] = 1;
        }
        for (int e : l.host_list)
            if (!is[(size_t) e]) vs[e] = l.n_slots + j++;
        for (const auto& x : cand) flow_push(c, l, mi, k, x.second, vs[x.second]);
        c.n_spec = (int) cand.size();
        c.it_spec += cand.size();
    }
    // (chunk k, layer il) routed (bounds b by virtual slot): the rest of its picks after its likely ones
    void flow_rest(Chunk& c, int k, int il, const std::vector<int>& b) {
        const Layer& l = lof(c, il);
        const int mi = flow_mi(c, l);
        const int* vs = member(l, mi).vs_pin;
        for (size_t i = c.spec_begin; i < c.spec_begin + (size_t) c.n_spec; ++i) {
            const int v = fit(c, i).v;
            if (b[(size_t) v + 1] == b[(size_t) v]) ++c.it_spec_unused;
        }
        for (int e : l.host_list) {   // (index order: their virtual slots ascending)
            const int v = vs[e];
            if (v >= l.n_slots + c.n_spec && b[(size_t) v + 1] > b[(size_t) v]) flow_push(c, l, mi, k, e, v);
        }
    }
    // the next (chunk, layer) of the device after (k, il) in the run's order (false: none)
    bool flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const {
        const auto it = std::find(c.fl_layers.begin(), c.fl_layers.end(), il);
        const size_t i = (size_t) (it - c.fl_layers.begin());
        if (c.lm) {   // (layer-major: the layer's next chunk, then the next layer's first)
            if (k + 1 < (int) c.fl_T.size()) {
                k2 = k + 1;
                il2 = il;
                return true;
            }
            if (i + 1 >= c.fl_layers.size()) return false;
            k2 = 0;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (i + 1 < c.fl_layers.size()) {
            k2 = k;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (k + 1 < (int) c.fl_T.size() && !c.fl_layers.empty()) {
            k2 = k + 1;
            il2 = c.fl_layers[0];
            return true;
        }
        return false;
    }
    // the reads (in order, each into its staging slot once that slot's last DMA is done) and the DMAs (in order, each
    // into its ring slot once that slot's last use is queued: the copy stream waits for it) as far as they can go now
    bool flow_pump(Chunk& c, std::string& err) {
        const size_t end = c.base + c.items.size();
        IoQueue::Job jobs[6];
        while (c.n_read < end) {
            Chunk::Item& it = fit(c, c.n_read);
            if (it.after >= 0) {   // layer-major: a slot's expert copied out of it once its reader is done; a slot
                // written again once the layer before's items are all read (none still copies out of it)
                bool wait = false;
                if (it.keep == nullptr)
                    wait = (size_t) it.after >= c.base && fit(c, (size_t) it.after).left.load(std::memory_order_acquire) != 0;
                else {
                    while (c.lc_chk < c.n_read &&
                           (c.lc_chk < c.base || fit(c, c.lc_chk).left.load(std::memory_order_acquire) == 0))
                        ++c.lc_chk;
                    wait = c.lc_chk <= (size_t) it.after;
                }
                if (wait) break;
            }
            if (it.stage >= 0) {
                const int64_t prev = c.stage_item[(size_t) it.stage];
                if (prev >= 0) {
                    if ((size_t) prev >= c.n_dma) break;
                    const cudaError_t q = cudaEventQuery(c.stage_ev[(size_t) it.stage]);
                    if (q == cudaErrorNotReady) break;
                    CK(q);
                }
                c.stage_item[(size_t) it.stage] = (int64_t) c.n_read;
                const Layer& l = lof(c, it.il);
                uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
                const int cs = copy_mode && it.ram != nullptr ? l.ram[(size_t) it.e] : -1;   // (a copy's slot)
                int nj = 0;
                for (int r = 0; r < 3; ++r) {
                    if (cs >= 0) {   // a copy: each matrix's NH parts of rows, side by side into staging
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X) {
                            IoQueue::Job& j = jobs[nj++];
                            j = IoQueue::Job{};
                            j.len = nb;
                            j.dst = sb + (size_t) X * nb;
                            j.left = &it.left;
                            j.bad = &it.bad;
                            j.mem = ram_part[(size_t) cs][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
                        }
                        sb += flow_region(r);
                        continue;
                    }
                    IoQueue::Job& j = jobs[nj++];
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) it.e * l.mx_stride[r] + mx_read_off(l, it.mi, r);
                    j.len = mx_read_len(l, r);
                    j.dst = sb;
                    j.left = &it.left;
                    j.bad = &it.bad;
                    j.mem = it.ram != nullptr ? it.ram + l.lay.trellis[r] + mx_read_off(l, it.mi, r) : nullptr;
                    j.map = pk.at(l.mx_file[r], j.off);
                    j.keep = it.keep != nullptr ? it.keep + l.lay.trellis[r] : nullptr;
                    sb += flow_region(r);
                }
                c.io->push(jobs, nj);
            }
            ++c.n_read;
        }
        const size_t R = (size_t) c.ring_n, dma0 = c.n_dma;
        struct Kick {   // (every way out: what was queued is submitted)
            cudaStream_t s;
            const size_t &a, b;
            ~Kick() {
                if (a != b) kick(s);
            }
        } kq{c.cs, c.n_dma, dma0};
        while (c.n_dma < c.n_read && c.n_dma < c.n_comp + R) {
            Chunk::Item& it = fit(c, c.n_dma);
            if (it.left.load(std::memory_order_acquire) != 0) break;
            if (it.bad.load(std::memory_order_acquire)) {
                err = "reading layer " + std::to_string(it.il) + " expert " + std::to_string(it.e) +
                      " from the model's files failed";
                return false;
            }
            const size_t r = c.n_dma % R;
            if (c.n_dma >= R) CK(cudaStreamWaitEvent(c.cs, c.ring_free[r], 0));
            const Layer& l = lof(c, it.il);
            uint8_t* dst = c.ring + r * c.ring_stride;
            const x3::ExpertLayout& lay = l.nm > 1 ? l.hlay : l.lay;
            for (int p = 0; p < 2; ++p)
                CK(cudaMemcpyAsync(dst + lay.trellis[p], it.src[p], mx_read_len(l, p), cudaMemcpyHostToDevice, c.cs));
            if (l.nm > 1) {   // (the down's part: in each of its rows the member's blocks)
                const int t2 = gg::expert_type(l.lay, 2);
                const size_t dr = gg::type_row_bytes(t2, g.ff), dh = gg::type_row_bytes(t2, g.ff / l.nm);
                CK(cudaMemcpy2DAsync(dst + lay.trellis[2], dh, it.src[2] + (size_t) it.mi * dh, dr, dh, (size_t) g.dim,
                                     cudaMemcpyHostToDevice, c.cs));
            } else
                CK(cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs));
            CK(cudaEventRecord(c.ring_fill[r], c.cs));
            if (it.stage >= 0) CK(cudaEventRecord(c.stage_ev[(size_t) it.stage], c.cs));
            c.stream_bytes += (double) unit(l);
            ++c.n_dma;
        }
        return true;
    }
    // layer L0 of chunk k routed (bounds b, the device's copy d_bounds): its streamed experts as they land, the VRAM
    // ones in shares between them (the device works while the reads run)
    bool flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b, std::string& err) {
        cudaStream_t s = d.st;
        const uint8_t* vbase = member(L0, mi).vram_base;   // (a member's: its part of each VRAM expert, L0's slots)
        if (c.spec_k != k || c.spec_il != L0.il) {
            err = "the prompt path's expert stream is out of step (layer " + std::to_string(L0.il) + ")";
            return false;
        }
        flow_rest(c, k, L0.il, b);
        const size_t lb = c.spec_begin, le = c.base + c.items.size(), n_lay = le - lb;
        int k2 = 0, il2 = 0;
        if (flow_next(c, k, L0.il, k2, il2)) flow_spec(c, k2, il2, c.fl_T[(size_t) k2]);
        const size_t R = (size_t) c.ring_n, bmax = std::max<size_t>(1, R / 2);
        int v_done = 0;
        while (c.n_comp < le) {
            // a batch's bounds are fixed by the stream alone (bmax items, never across the ring's end), never by what
            // has landed: the experts' outputs are summed batch after batch (the VRAM ones' shares between them), so
            // the sums come out the same on every run whatever the reads' timing
            const size_t i0 = c.n_comp, i1 = std::min({le, i0 + bmax, (i0 / R + 1) * R});
            if (!flow_pump(c, err)) return false;
            if (c.n_dma < i1) {   // not all of the batch landed: wait for a read, or for a staging slot's copy
                const double tw = now_ms();
                kick(s);
                if (c.n_read > c.n_dma) c.io->wait_any(500);
                else {
                    const Chunk::Item& it = fit(c, c.n_read);
                    if (it.stage >= 0) CK(cudaEventSynchronize(c.stage_ev[(size_t) it.stage]));
                }
                c.wait_ms += now_ms() - tw;
                continue;
            }
            const int n = (int) (i1 - i0);
            int* bc = c.bc_pin + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
            int* bcd = c.bc_dev + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
            ++c.n_batch;
            // (the items' virtual slots ascend and the slots between them have no rows: their bounds side by side)
            for (int m = 0; m < n; ++m) bc[m] = b[(size_t) fit(c, i0 + (size_t) m).v];
            bc[n] = b[(size_t) fit(c, i1 - 1).v + 1];
            if (bc[n] > bc[0]) {
                CK(cudaStreamWaitEvent(s, c.ring_fill[(i1 - 1) % R], 0));
                CK(cudaMemcpyAsync(bcd, bc, (size_t) (n + 1) * 4, cudaMemcpyHostToDevice, s));
                if (!run_set(d, c, L0, c.ring + (i0 % R) * c.ring_stride, 0, c.ring_stride, bcd, bc, 0, n, T, err))
                    return false;
            }
            for (size_t i = i0; i < i1; ++i) CK(cudaEventRecord(c.ring_free[i % R], s));
            kick(s);
            c.n_comp = i1;
            if (!flow_pump(c, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (int64_t) (c.n_comp - lb) / (int64_t) std::max<size_t>(1, n_lay));
            if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        while (c.base < c.n_comp && c.items.size() > 4096) {   // (done with: read, copied and used)
            c.items.pop_front();
            ++c.base;
        }
        return true;
    }

    // the experts [i0, i1) of a sorted set (blobs at wbase + i * stride; bounds b on the host, d_bounds on the device)
    // over the chunk's rows, in pieces of at most C rows, each added into acc in pick order
    bool run_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride, const int* d_bounds,
                 const int* b, int i0, int i1, int T, std::string& err) {
        const int D = g.dim, FF = l.ffn;
        while (i0 < i1) {
            int j = i0 + 1;
            while (j < i1 && b[j + 1] - b[i0] <= c.C) ++j;
            const int a = b[i0], nrows = b[j] - a;
            int max_rows = 0;
            for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, b[i + 1] - b[i]);
            if (nrows > 0) {
                x3::SetArgs sa;
                sa.wbase = wbase + (size_t) (i0 - wb0) * stride;   // (wbase: set member wb0's blob)
                sa.stride = stride;
                sa.lay = l.nm > 1 ? l.hlay : l.lay;
                sa.n_exp = j - i0;
                sa.bounds = d_bounds + i0;
                sa.row0 = a;
                sa.max_rows = max_rows;
                sa.n_embd = D;
                sa.n_ff = FF;
                sa.limit = g.lim;
                // (temporary: each set launch synced on the device before the next is queued, so a fault is that
                // kernel's and names its layer, device and piece)
                auto rp_set_sync = [&](const char* what) -> bool {
                    if (std::getenv("DS4_RP_CHECK") == nullptr) return true;
                    const cudaError_t e = cudaStreamSynchronize(d.st);
                    if (e != cudaSuccess) {
                        char wb[32];
                        std::snprintf(wb, sizeof wb, "%p", (const void*) sa.wbase);
                        err = std::string("rowsplit ") + what + " (layer " + std::to_string(l.il) + " dev " +
                              std::to_string(l.dev) + " a " + std::to_string(a) + " nrows " + std::to_string(nrows) +
                              " nexp " + std::to_string(sa.n_exp) + " maxr " + std::to_string(max_rows) + " wbase " +
                              wb + "): " + cudaGetErrorString(e);
                        return false;
                    }
                    return true;
                };
                if (std::getenv("DS4_RP_CHECK") != nullptr) {   // (temporary: the row split's set arguments, validated)
                    static int lines = 0;
                    CK(cudaStreamSynchronize(c.cs));   // (temporary: a copy stream's fault named before the launch)
                    CK(cudaStreamSynchronize(d.st));
                    std::vector<int> db((size_t) sa.n_exp + 1), rt((size_t) nrows);
                    CK(cudaMemcpy(db.data(), d_bounds + i0, ((size_t) sa.n_exp + 1) * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(rt.data(), c.row_tok + a, (size_t) nrows * 4, cudaMemcpyDeviceToHost));
                    int bads = 0;
                    char why[256] = "";
                    if (db[0] != a) {
                        ++bads;
                        std::snprintf(why, sizeof why, "bounds[0] %d != a %d", db[0], a);
                    }
                    for (int q = 0; q <= sa.n_exp && bads < 3; ++q)
                        if (db[q] < a || db[q] > a + nrows || (q > 0 && db[q] < db[q - 1])) {
                            ++bads;
                            std::snprintf(why, sizeof why, "bounds[%d] %d outside [%d, %d] (prev %d)", q, db[q], a,
                                          a + nrows, q > 0 ? db[q - 1] : -1);
                        }
                    for (int q = 0; q < nrows && bads < 3; ++q)
                        if (rt[q] < 0 || rt[q] >= T) {
                            ++bads;
                            std::snprintf(why, sizeof why, "row_tok[%d] %d outside [0, %d)", q, rt[q], T);
                        }
                    const bool ring_ok = wbase >= c.ring &&
                                         wbase + (size_t) sa.n_exp * stride <= c.ring + (size_t) c.ring_n * stride;
                    if (lines < 400)
                        std::fprintf(stderr,
                                     "ds4 rpcheck: layer %d dev %d T %d C %d a %d nrows %d nexp %d maxr %d"
                                     " wbase %p sa.wbase %p wb0 %d ring %p ring_n %d stride %zu ring_ok %d"
                                     " h16 %p x %p y %p%s%s\n"
                                     "ds4 rpcheck:   fmt %d tr %zu %zu %zu ff %d l.ffn %d nm %d D %d dense %zu"
                                     " bounds %p row0 %d\n",
                                     l.il, l.dev, T, c.C, a, nrows, sa.n_exp, max_rows, (const void*) wbase,
                                     (const void*) sa.wbase, wb0, (const void*) c.ring, c.ring_n, stride,
                                     ring_ok ? 1 : 0, (const void*) c.h16, (const void*) c.x, (const void*) c.y,
                                     bads ? " BAD: " : "", why,
                                     (int) sa.lay.fmt, sa.lay.trellis[0], sa.lay.trellis[1], sa.lay.trellis[2], FF,
                                     l.ffn, l.nm, D, l.dense_bytes, (const void*) sa.bounds, sa.row0);
                    ++lines;
                    if (bads) {
                        err = std::string("rp check: ") + why;
                        return false;
                    }
                }
                if (sa.lay.fmt == gg::kFmtMXFP4) {
                    bool done = false;
#if defined(STRATA_DS4_MMQ)
                    // --chunk-mmq: the same packed MXFP4 blocks through MMQ's tensor-core products (the activations
                    // block-quantized, the window's bounds made relative on device); the dump path reads c.h16, so
                    // it keeps the GGUF kernels
                    if (mmq_do() && dump_y == nullptr) {
                        if (!d.mmq_tried) {
                            d.mmq_tried = true;
                            d.mmq = mmq::create(d.id, d.st, mmq_rows(), D, FF);
                        }
                        if (d.mmq != nullptr) {
                            mmq::Call mc;
                            mc.sa = &sa;
                            mc.hb = b;
                            mc.x = c.x;
                            mc.ldx = D;
                            mc.row_tok = c.row_tok + a;
                            mc.y = c.y;
                            mc.ldy = D;
                            mc.i0 = i0;
                            mc.i1 = j;
                            done = mmq::run(d.mmq, mc, d.st);
                        }
                    }
#endif
                    if (!done) {
                        gg::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                        if (!rp_set_sync("set_gate_up")) return false;
                        gg::set_down(sa, c.h16, FF, c.y, D, d.st);
                        if (!rp_set_sync("set_down")) return false;
                    }
                } else {
                    x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    if (!rp_set_sync("x3 set_gate_up")) return false;
                    x3::set_down(sa, c.h16, FF, c.y, D, d.st);
                    if (!rp_set_sync("x3 set_down")) return false;
                }
                if (dump_y != nullptr && (dump_y_layer < 0 || l.il == dump_y_layer)) {
                    std::vector<int> bb(b + i0, b + j + 1);
                    std::vector<uint16_t> hv((size_t) nrows * FF);
                    std::vector<float> yv((size_t) nrows * D);
                    CK(cudaStreamSynchronize(d.st));
                    CK(cudaMemcpy(hv.data(), c.h16, hv.size() * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(yv.data(), c.y, yv.size() * 4, cudaMemcpyDeviceToHost));
                    std::lock_guard<std::mutex> lk(dump_mu);
                    const int64_t hdr[6] = {l.il, a, nrows, j - i0, wbase == c.ring || (wbase > c.ring &&
                                            wbase < c.ring + (size_t) c.ring_n * c.ring_stride) ? 1 : 0, max_rows};
                    std::fwrite(hdr, 8, 6, dump_y);
                    std::fwrite(bb.data(), 4, bb.size(), dump_y);
                    std::fwrite(hv.data(), 2, hv.size(), dump_y);
                    std::fwrite(yv.data(), 4, yv.size(), dump_y);
                    std::fflush(dump_y);
                }
                moe_combine(c.acc, c.y, c.pos, c.rw, T, l.k_exp, a, a + nrows, D, d.st);
                if (std::getenv("DS4_RP_CHECK") != nullptr) {   // (temporary: the combine's launch, checked at once)
                    const cudaError_t e = cudaGetLastError();
                    if (e != cudaSuccess) {
                        err = "rowsplit after moe_combine (layer " + std::to_string(l.il) + " dev " +
                              std::to_string(l.dev) + " a " + std::to_string(a) + " nrows " + std::to_string(nrows) +
                              "): " + cudaGetErrorString(e);
                        return false;
                    }
                }
            }
            i0 = j;
        }
        (void) err;
        return true;
    }

    // one layer's dense weights, its norms and (state) the small per-layer buffers it owns: the body load_dense runs
    // for every member of every layer it holds, and the row-split prefill's group swap for a replica (state = false:
    // the replica's ring and compressor state were allocated once, at load - see rp_setup)
    bool load_layer_dense(Dev& d, Layer& l, int nm, int mi, bool state, std::string& err, size_t* acct) {
        // (a group's member: its share of the heads (wq_b columns, the sinks), of wo_a's groups and wo_b's rows,
        // of the shared expert's FFN; the rest whole)
        const int na = g.heads / std::max(1, l.ahn), ha = l.ah0 / std::max(1, l.ahn);
        const std::string& P = l.name;
        const std::string A = P + "attn.";
        bool ok = mat(d, A + "wq_a", l.wq_a, err, acct) &&
                  (l.ahn == 0 || mat(d, A + "wq_b", l.wq_b, err, acct, ha, na, true)) &&
                  mat(d, A + "wkv", l.wkv, err, acct) &&
                  (l.ahn == 0 || mat(d, A + "wo_b", l.wo_b, err, acct, ha, na, false)) &&
                  f32(d, A + "q_norm.weight", l.q_norm, err, acct, (size_t) g.q_lora) &&
                  f32(d, A + "kv_norm.weight", l.kv_norm, err, acct, (size_t) g.hd) &&
                  f32(d, A + "attn_sink", l.sink, err, acct, (size_t) g.heads) &&
                  f32(d, P + "attn_norm.weight", l.attn_norm, err, acct, (size_t) g.dim) &&
                  f32(d, P + "ffn_norm.weight", l.ffn_norm, err, acct, (size_t) g.dim) &&
                  f32(d, P + "hc_attn_fn", l.hca_fn, err, acct) && f32(d, P + "hc_attn_scale", l.hca_scale, err, acct) &&
                  f32(d, P + "hc_attn_base", l.hca_base, err, acct) && f32(d, P + "hc_ffn_fn", l.hcf_fn, err, acct) &&
                  f32(d, P + "hc_ffn_scale", l.hcf_scale, err, acct) &&
                  f32(d, P + "hc_ffn_base", l.hcf_base, err, acct) &&
                  f32(d, P + "ffn.gate.weight", l.gate_w, err, acct, (size_t) l.n_exp * g.dim) &&
                  f32(d, P + "ffn.gate.bias", l.gate_bias, err, acct, (size_t) l.n_exp) &&
                  (l.mtp || img_tok < 0 || pk.get(P + "ffn.gate.bias_vl") == nullptr ||
                   f32(d, P + "ffn.gate.bias_vl", l.gate_bias_vl, err, acct, (size_t) l.n_exp)) &&
                  mat(d, P + "ffn.shared_experts.w1", l.s1, err, acct, mi, nm, true, l.il % nm) &&
                  mat(d, P + "ffn.shared_experts.w2", l.s2, err, acct, mi, nm, false, l.il % nm) &&
                  mat(d, P + "ffn.shared_experts.w3", l.s3, err, acct, mi, nm, true, l.il % nm);
        if (!ok) return false;
        l.sink += (size_t) l.ah0;
        for (int gi = 0; gi < l.ogn; ++gi)
            if (!mat(d, A + "wo_a.slice." + std::to_string(l.og0 + gi), l.wo_a[gi], err, acct)) return false;
        if (l.kv_src) {
            const std::string C = A + "compressor.";
            if (!mat(d, C + "wkv", l.c_wkv, err, acct) || !f32(d, C + "norm.weight", l.c_norm, err, acct)) return false;
            if (l.ratio > 1 && !mat(d, C + "wgate", l.c_wgate, err, acct)) return false;
            if (state && l.ratio > 1 && (!dalloc(d, l.c_kv, (size_t) l.ratio * g.hd, err) ||
                                          !dalloc(d, l.c_score, (size_t) l.ratio * g.hd, err)))
                return false;
        }
        if (l.ix_src) {
            const std::string I = A + "indexer.";
            if (!mat(d, I + "wq_b", l.i_wq_b, err, acct) || !f16raw(d, I + "weights_proj.weight", l.i_wproj, err, acct))
                return false;
            if (l.owns_k && (!mat(d, I + "wk", l.i_wk, err, acct) || !f32(d, I + "k_norm.weight", l.ik_norm, err, acct)))
                return false;
        }
        if (l.eg >= 0) {
            if (!mat(d, P + "engram.wkv", l.eg_wkv, err, acct)) return false;
            if (pk.get(P + "engram.gate.weight") != nullptr) {   // a GGUF: q_weight * k_weight, folded already
                if (!f32(d, P + "engram.gate.weight", l.eg_w, err, acct, (size_t) g.hc * g.dim)) return false;
            } else {
                float *qw = nullptr, *kw = nullptr;
                if (!f32(d, P + "engram.q_weight", qw, err, acct) || !f32(d, P + "engram.k_weight", kw, err, acct))
                    return false;
                std::vector<float> a((size_t) g.hc * g.dim), b(a.size());
                CK(cudaMemcpy(a.data(), qw, a.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), kw, b.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];   // only ever used as a product
                CK(cudaMemcpy(qw, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
                l.eg_w = qw;
            }
        }
        if (state) {
            if (!dalloc(d, l.win, (size_t) g.win * win_rb(), err, &d.caches)) return false;
            if (g.n_mtp > 0 && !l.mtp) {
                if (!dalloc(d, l.ring_bak, (size_t) kWin * win_rb(), err)) return false;
                if (l.kv_src && l.ratio > 1 && (!dalloc(d, l.c_snap, (size_t) 2 * l.ratio * g.hd, err) ||
                                                !dalloc(d, l.c_rows, (size_t) 2 * kWin * g.hd, err)))
                    return false;
            }
            if (!dalloc(d, l.vram_tab, (size_t) l.n_exp, err, acct)) return false;
        }
        return true;
    }

    bool load_dense(int di, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        size_t* acct = &d.weights;
        for (Layer& l0 : L)
          for (int mi = 0; mi < l0.nm; ++mi) {
            Layer& l = member(l0, mi);
            if (l.dev != di) continue;
            if (!load_layer_dense(d, l, l0.nm, mi, true, err, acct)) return false;
          }
        if (d.grp == devs[(size_t) head_dev].grp) {   // the head: each member of its group a share of the columns
            const int hn = head_nm();
            if (!mat(d, "head", d.head, err, acct, d.mi, hn, true) || !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim))
                return false;
            if (g.n_mtp > 0 && !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight", d.d_norm, err, acct, (size_t) g.dim))
                return false;
            if (!dalloc(d, d.logits_part, (size_t) kWin * ((g.vocab / 128 + hn - 1) / hn * 128), err)) return false;
            if (g.n_mtp > 0 && hn > 1) {   // the markov steps over the group (mk_split): this member's rows of the head
                const Tensor* th = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.head.weight");
                const int R = g.d_rank;
                head_cols(hn, d.mi, d.mk_v0, d.mk_vn);   // (its share of the head's columns: the same rows)
                // F16 as the pack holds it, or a GGUF drafter's BF16 rounded to F16 as f16raw rounds d_mk_head (the same
                // values as the lead's whole rows); another type: no mk_split (the lead's whole rows)
                if (th != nullptr && (th->dtype == "F16" || th->dtype == "BF16") && th->bytes == (uint64_t) g.vocab * R * 2) {
                    std::vector<uint16_t> rows((size_t) d.mk_vn * R);
                    if (!pk.read_at(th->file, th->off + (uint64_t) d.mk_v0 * R * 2, rows.size() * 2, rows.data())) {
                        err = "the markov head: read failed";
                        return false;
                    }
                    if (th->dtype == "BF16")
                        for (uint16_t& h : rows) h = strata::kernels::f16_from_f32(bf16_to_f32(h));
                    if (!dalloc(d, d.mk_share, rows.size(), err, acct) || !dalloc(d, d.mk_erow, (size_t) R, err) ||
                        !dalloc(d, d.mk_ints, (size_t) kWin + 2, err))
                        return false;
                    CK(cudaMemcpy(d.mk_share, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
                }
            }
        }
        if (di == head_dev) {
            head = d.head;
            norm_w = d.norm_w;
            d_norm = d.d_norm;
            if (g.n_mtp > 0) {
                const std::string Z = "mtp." + std::to_string(g.n_mtp - 1) + ".";
                if (!f32(d, Z + "confidence_head.proj.weight", d_conf, err, acct, (size_t) (g.dim + g.d_rank)))
                    return false;
                const Tensor* te = pk.get(Z + "markov_head.embed.weight");
                const Tensor* th = pk.get(Z + "markov_head.head.weight");
                if (te == nullptr || th == nullptr || te->dtype != "BF16" || (th->dtype != "F16" && th->dtype != "BF16")) {
                    err = Z + "markov_head: not the BF16 embed and F16 / BF16 head expected";
                    return false;
                }
                if (!dalloc(d, d_mk_embed, te->bytes / 2, err, acct) || !upload(d, *te, d_mk_embed, err) ||
                    !f16raw(d, Z + "markov_head.head.weight", d_mk_head, err, acct))
                    return false;
            }
        }
        if (g.n_mtp > 0 && di == d_dev) {
            if (!mat(d, "mtp.0.main_proj", d_main_proj, err, acct) ||
                !f32(d, "mtp.0.main_norm.weight", d_main_norm, err, acct, (size_t) g.dim))
                return false;
        }
        return true;
    }

    void set_vm(Layer& l, int e, int slot) {
        uint8_t* b = slot_ptr(l, slot);   // (the lead alone: a group's members take the table path)
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r) {
                x3::Mat& m = l.vm[(size_t) e][(size_t) r];
                m = x3::Mat{};
                m.K = gg::expert_type(l.lay, r);   // (a low-bit type: its GEMV, ds4_iq's)
                m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
                m.raw = b + l.lay.trellis[r];
                m.k = r < 2 ? g.dim : g.ff;
                m.n = r < 2 ? g.ff : g.dim;
            }
            return;
        }
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m.trellis = (const uint32_t*) (b + l.lay.trellis[r]);
            m.suh = (const uint16_t*) (b + l.lay.suh[r]);
            m.svh = (const uint16_t*) (b + l.lay.svh[r]);
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.K = l.lay.K[r];
            m.cb = 2;
            m.ld = 0;
        }
    }
    // an expert's VRAM blob -> its CPU-lane part X, and its NH parts -> the blob (the cache's swaps)
    void blob_to_part(const Layer& l, const uint8_t* blob, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = blob + l.lay.trellis[r];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], blob + l.lay.suh[r], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }
    void parts_to_blob(const Layer& l, const std::array<uint8_t*, 4>& parts, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            uint8_t* tr = blob + l.lay.trellis[r];
            for (int X = 0; X < NH; ++X)
                for (int sl = 0; sl < k / 16; ++sl)
                    std::memcpy(tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                                parts[(size_t) X] + l.hl.trellis[r] + (size_t) sl * nth * tile, (size_t) nth * tile);
            std::memcpy(blob + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, parts[(size_t) X] + l.hl.svh[r],
                            (size_t) l.hl.n[r] * 2);
        }
    }

    // an expert's blob -> member m's part (gate / up columns [ff0, ff0 + ffn), down rows of the same), and back
    void blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_blob_to_half(l, blob, m, dst);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(dst + x.hlay.trellis[r] + sl * nth * tile,
                            blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile, nth * tile);
            std::memcpy(dst + x.hlay.suh[r], blob + l.lay.suh[r], (size_t) g.dim * 2);
            std::memcpy(dst + x.hlay.svh[r], blob + l.lay.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(dst + x.hlay.trellis[2], blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, (size_t) (fn / 16) * row);
        std::memcpy(dst + x.hlay.suh[2], blob + l.lay.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        std::memcpy(dst + x.hlay.svh[2], blob + l.lay.svh[2], (size_t) g.dim * 2);
    }
    void half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_half_to_blob(l, src, m, blob);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile,
                            src + x.hlay.trellis[r] + sl * nth * tile, nth * tile);
            if (m == 0) std::memcpy(blob + l.lay.suh[r], src + x.hlay.suh[r], (size_t) g.dim * 2);
            std::memcpy(blob + l.lay.svh[r] + (size_t) f0 * 2, src + x.hlay.svh[r], (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, src + x.hlay.trellis[2], (size_t) (fn / 16) * row);
        std::memcpy(blob + l.lay.suh[2] + (size_t) f0 * 2, src + x.hlay.suh[2], (size_t) fn * 2);
        if (m == 0) std::memcpy(blob + l.lay.svh[2], src + x.hlay.svh[2], (size_t) g.dim * 2);
    }

    bool place_experts(std::string& err) {
        // the arenas' offsets start at 0 again: the drafter's banks take the first of each device's, the trunk's
        // lend banks the rest (see the pass after the layers' allocations)
        for (Dev& d : devs) d.arena_off = 0;
        // (Windows, an integrated GPU: what its experts may commit - commit_room - and what that kept off the GPU)
        double room = 1e300, kept = 0;
        for (const Dev& d : devs)
            if (commit_charged(d)) room = commit_room();
        const double room0 = room;
        // per group: what its tightest member has left (each member holds its part of every VRAM expert of the
        // group's layers, its unit bytes a slot - counted at the members' mean: an uneven split's odd blocks go round
        // the members layer by layer, so each member's total comes to about the mean's)
        auto ucount = [](const Layer& l) {
            double s = 0;
            for (int m = 0; m < l.nm; ++m) s += (double) unit(member(l, m));
            return s / l.nm;
        };
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            double budget = 1e300;
            bool charged = false;
            for (int m : groups[gi].mem) {
                cudaSetDevice(devs[(size_t) m].id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                budget = std::min(budget, std::max(0.0, ((double) fr - (double) resv[(size_t) m])) * o.vram_experts);
                charged = charged || commit_charged(devs[(size_t) m]);
            }
            if (charged && room < 1e300) {   // (each member's slots charged: the group's share of the room)
                double all = 0;
                for (const Layer& l : L)
                    if (l.grp == (int) gi) all += (double) unit(l) * l.n_exp;
                const double nm = (double) groups[gi].mem.size(), capped = std::min(budget, room / nm);
                kept += std::max(0.0, std::min(budget, all) - std::min(capped, all)) * nm;
                budget = capped;
                room = std::max(0.0, room - std::min(budget, all) * nm);
            }
            // the drafter's experts first, out of the prompt arena (not the free budget: the arena is already
            // allocated, so charging them here would leave the bytes they used to take from free unspent).  Their
            // banks are their device's arena's first; the rest is lent to the trunk (the pass after the allocations).
            // A lane kept in host memory (nothing to read it back from when a prompt takes the arena) keeps the old
            // way instead: from free, fixed slots.
            double budget_t = budget;
            for (Layer& l : L) {
                if (l.grp != (int) gi || !l.mtp) continue;
                bool arena_ok = o.expert_lend && o.expert_cache && lane_file(l);
                for (int m = 0; m < l.nm && arena_ok; ++m)
                    arena_ok = devs[(size_t) member(l, m).dev].arena_bytes > 0;   // (no chunk buffers: no arena)
                if (!arena_ok) {
                    l.n_vram = std::min(l.n_exp, (int) std::floor(budget_t / (double) ucount(l)));
                    budget_t -= (double) l.n_vram * ucount(l);
                    continue;
                }
                int n = l.n_exp;
                for (int m = 0; m < l.nm; ++m) {
                    const Dev& dm = devs[(size_t) member(l, m).dev];
                    const size_t left = dm.arena_bytes > dm.arena_off ? dm.arena_bytes - dm.arena_off : 0;
                    n = std::min(n, (int) (left / align256(unit(l))));
                }
                n = std::max(0, n);
                l.lend = true;
                l.n_perm = 0;
                l.n_lend = n;
                if (o.expert_cache && n >= 2 && n < l.n_exp) {   // (one slot kept for its swaps)
                    l.n_vram = n - 1;
                    l.spare_vram = n - 1;
                } else {
                    l.n_vram = n;
                    l.spare_vram = -1;
                }
                for (int m = 0; m < l.nm; ++m) {
                    Layer& x = member(l, m);
                    Dev& dm = devs[(size_t) x.dev];
                    x.vram_base = x.lend_base = dm.arena + dm.arena_off;
                    x.n_perm = 0;
                    x.n_lend = n;
                    x.lend = true;
                    dm.arena_off = align256(dm.arena_off + (size_t) n * unit(l));
                    dm.experts += (size_t) n * unit(l);
                }
                l.n_slots = 0;   // (the prompt layout's count: the drafter's slots exist between prompts only)
            }
            double want = 0;
            for (const Layer& l : L)
                if (l.grp == (int) gi && !l.mtp) want += (double) ucount(l) * l.n_exp;
            const double frac = want > 0 ? std::min(1.0, std::max(0.0, budget_t) / want) : 0.0;
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.mtp) continue;
                l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
                // never more than fits: the floor of each layer's share can still round up the total
                while (l.n_vram > 0 && (double) ucount(l) * l.n_vram > budget_t) --l.n_vram;
            }
            double used = 0;
            for (Layer& l : L)
                if (l.grp == (int) gi && !l.mtp) used += (double) ucount(l) * l.n_vram;
            // what the floors left: whole experts, layer by layer
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.mtp) continue;
                while (l.n_vram < l.n_exp && used + (double) ucount(l) <= budget_t) {
                    ++l.n_vram;
                    used += (double) ucount(l);
                }
            }
        }
        if (kept > 0) {   // the commit kept experts off the GPU: say so, and the page file that keeps them on it
            const os::Commit c = os::commit();
            const double gib = (double) (1ull << 30);
            const double pf = c.limit != ~uint64_t{0} && c.ram != ~uint64_t{0} && c.limit > c.ram ? (double) (c.limit - c.ram) : 0.0;
            commit_kept = kept;
            pagefile_hint = std::ceil((pf + kept) / gib);
            std::fprintf(stderr,
                         "ds4: WARNING: %.1f GB of experts stay off the GPU - Windows charges the GPU's memory to its "
                         "commit (RAM + page file) and can commit only %.1f GB more for them (%.0f GB kept for the "
                         "other programs). A page file of %.0f GB (%.0f MB; now %.0f GB) keeps them on the GPU: System "
                         "> About > Advanced system settings > Performance > Advanced > Virtual memory, then restart\n",
                         kept / 1e9, room0 / 1e9, commit_margin() / 1e9, pagefile_hint, pagefile_hint * 1024.0, pf / gib);
        } else if (room0 < 1e300 && o.verbose) {
            std::fprintf(stderr, "ds4: Windows' commit has room for the GPU's experts (%.1f GB beside %.0f GB for the other "
                                 "programs)\n", room0 / 1e9, commit_margin() / 1e9);
        }
        // DS4_PASS_LEND: an encoder layer's bank and a decoder-tail layer's on one device as ONE block, the tail's above
        // the encoder's, so the encoder layer's slot numbering runs on into the tail's bank - a prompt's pass 1 gives the
        // encoder layer the whole block, pass 2 the tail (pl_borrow).  The same slots and bytes as the plain way below;
        // nothing changes when the flag is off
        pl_pairs.clear();
        std::vector<int> pl_block_of(L.size(), -1);   // layer -> its block in pl_pairs (-1: a bank of its own)
        if (pl_do() && o.expert_lend && o.expert_cache) {
            for (size_t di = 0; di < devs.size(); ++di) {
                std::vector<int> enc, tail;
                for (const Layer& l : L) {
                    if (l.mtp || l.nm != 1 || l.dev != (int) di || l.lay.fmt != gg::kFmtMXFP4 || !lane_file(l)) continue;
                    if (l.n_vram <= 0 || l.n_vram >= l.n_exp) continue;
                    (l.il <= g.cand_src ? enc : tail).push_back(l.il);
                }
                for (size_t i = 0; i < enc.size() && i < tail.size(); ++i) {
                    const Layer& e = L[(size_t) enc[i]];
                    const Layer& t = L[(size_t) tail[i]];
                    const int ne = std::min(e.n_vram, e.n_exp), nt = std::min(t.n_vram, t.n_exp);
                    // (each expert's blob is its layer's own: the two banks sit one above the other, the tail's at the
                    // next 256 bytes after the encoder's - a borrow takes the block's whole length in the borrower's unit)
                    const size_t ue = unit(e), ut = unit(t);
                    const size_t off_t = align256((size_t) ne * ue), blk = off_t + (size_t) nt * ut;
                    const int n_e = std::min(e.n_exp, (int) (blk / ue)), n_t = std::min(t.n_exp, (int) (blk / ut));
                    if (n_e <= ne || n_t <= nt) continue;   // (no room lent: the plain banks below)
                    Dev& d = devs[di];
                    cudaSetDevice(d.id);
                    void* v = nullptr;
                    if (cudaMalloc(&v, blk) != cudaSuccess) continue;   // (the plain banks below)
                    d.allocs.push_back(v);
                    d.experts += blk;
                    pl_block_of[(size_t) enc[i]] = pl_block_of[(size_t) tail[i]] = (int) pl_pairs.size();
                    pl_pairs.push_back({enc[i], tail[i], (uint8_t*) v, off_t, n_e, n_t, ne, nt});
                }
            }
            if (!pl_pairs.empty() && o.verbose) {
                size_t be = 0, bt = 0;
                for (const PlPair& b : pl_pairs) {
                    be += (size_t) b.ne * unit(L[(size_t) b.enc]);
                    bt += (size_t) b.nt * unit(L[(size_t) b.tail]);
                }
                std::fprintf(stderr, "ds4 pass-lend: %zu bank pairs: %.1f GB of encoder banks, %.1f GB of tail banks "
                                     "(each pass lends the one it does not run)\n",
                             pl_pairs.size(), be / 1e9, bt / 1e9);
            }
        }
        for (Layer& l : L) {
            // the hottest experts of the profile first, then by index
            std::vector<int> order((size_t) l.n_exp);
            std::iota(order.begin(), order.end(), 0);
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
            l.slot.assign((size_t) l.n_exp, -1);
            l.host.assign((size_t) l.n_exp, nullptr);
            // (the drafter's banks are the arena's - the MTP pass above: nothing to allocate, and the prompt layout
            // has them offline until the first refill after a prompt)
            const bool arena_bank = l.mtp && l.lend;
            if (arena_bank) {
                l.n_slots = 0;
            } else {
                // a layer with experts on both sides keeps one VRAM slot free for the cache's swaps
                l.n_slots = std::min(l.n_vram, l.n_exp);
                if (o.expert_cache && l.n_slots >= 2 && l.n_slots < l.n_exp) {   // (every expert fits: nothing to swap)
                    l.n_vram = l.n_slots - 1;
                    l.spare_vram = l.n_slots - 1;
                } else {
                    l.n_vram = l.n_slots;
                    l.spare_vram = -1;
                }
                for (int i = 0; i < l.n_vram; ++i) l.slot[(size_t) order[(size_t) i]] = i;
                for (int m = 0; m < l.nm && l.n_slots > 0; ++m) {
                    Layer& x = member(l, m);
                    Dev& d = devs[(size_t) x.dev];
                    cudaSetDevice(d.id);
                    const int bi = pl_block_of[(size_t) l.il];
                    if (bi >= 0) {   // (the block it shares with its pair: place_experts' pass above, bytes and count)
                        const PlPair& b = pl_pairs[(size_t) bi];
                        x.vram_base = b.base + (l.il == b.tail ? b.tail_off : 0);
                        continue;
                    }
                    void* v = nullptr;
                    if (cudaMalloc(&v, (size_t) l.n_slots * unit(x)) != cudaSuccess) {
                        err = "cudaMalloc of layer " + std::to_string(l.il) + "'s " + std::to_string(l.n_slots) +
                              " experts on CUDA" + std::to_string(d.id) + " failed";
                        return false;
                    }
                    d.allocs.push_back(v);
                    d.experts += (size_t) l.n_slots * unit(x);
                    x.vram_base = (uint8_t*) v;
                }
                l.n_perm = l.n_slots;   // (the prompt layout's count; the lend pass below adds the arena's)
                for (int m = 0; m < l.nm; ++m) member(l, m).n_perm = l.n_slots;
            }
            l.heat.assign((size_t) l.n_exp, 0.0f);
            {   // the profile as the starting heat: its counts scaled to ~512 hits a layer
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                const double sc = sum > 0 ? 512.0 / sum : 0.0;
                for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) (p[e] * sc);
            }
            const int n_host = l.n_exp - l.n_vram;
            {   // the CPU lane's parts of an expert (in place: the whole expert, its pieces where the file has them)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : g.ff, nf = r < 2 ? g.ff : g.dim, n = nf / NH;
                    l.hl.k[r] = k;
                    l.hl.n[r] = n;
                    l.hl.trellis[r] = file_experts ? l.piece_src[r][0] : off;
                    off += align256((size_t) (k / 16) * (n / 16) * 32 * l.lay.K[r]);
                    l.hl.suh[r] = file_experts ? l.piece_src[r][1] : off;
                    off += align256((size_t) k * 2);
                    l.hl.svh[r] = file_experts ? l.piece_src[r][2] : off;
                    off += align256((size_t) n * 2);
                }
                l.hl.bytes = off;
            }
            l.hh.assign((size_t) l.n_exp, {nullptr, nullptr, nullptr, nullptr});
            if (l.lay.fmt == gg::kFmtMXFP4) {   // a GGUF's: read in place from the file's mapping
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = (uint8_t*) mx_mat(l, e, 0);
            } else if (file_experts) {   // every expert's file mapped (the cache's swaps hand VRAM experts back to the files)
                for (int e = 0; e < l.n_exp; ++e)
                    if (pk.map(l.region_file[(size_t) e]) == nullptr) {
                        err = "mapping " + pk.files[(size_t) l.region_file[(size_t) e]] + " failed";
                        return false;
                    }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) {
                        l.hh[(size_t) e][0] = file_part(l, e);
                        l.host[(size_t) e] = l.hh[(size_t) e][0];
                    }
            } else if (n_host > 0) {
                const int n_parts = n_host + (l.spare_vram >= 0 ? 1 : 0);   // (+ the cache's spare)
                const size_t bytes = (size_t) n_parts * l.hl.bytes;
                for (int X = 0; X < NH; ++X) {
                    void* m = node_map(bytes, X, NH);
                    if (m == nullptr) {
                        err = "host memory for layer " + std::to_string(l.il) + "'s " + std::to_string(n_host) +
                              " CPU experts (" + std::to_string(bytes >> 20) + " MiB on node " + std::to_string(X) + ")";
                        return false;
                    }
                    host_maps.emplace_back(m, bytes);
                    int hi = 0;
                    for (int e = 0; e < l.n_exp; ++e)
                        if (l.slot[(size_t) e] < 0) l.hh[(size_t) e][(size_t) X] = (uint8_t*) m + (size_t) (hi++) * l.hl.bytes;
                    if (l.spare_vram >= 0) l.spare_host[(size_t) X] = (uint8_t*) m + (size_t) hi * l.hl.bytes;
                }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = l.hh[(size_t) e][0];
            }
            l.vm.assign((size_t) l.n_exp, {});
            if (l.nm == 1)   // (the host-routed window path's views: a group's members take the device-routed path)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0) set_vm(l, e, l.slot[(size_t) e]);
            if (!halloc(l.tab_pin, (size_t) 2 * l.nm, err)) return false;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) slot_ptr(x, l.slot[(size_t) e]);
                cudaSetDevice(devs[(size_t) x.dev].id);
                CK(cudaMemcpy(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice));
            }
        }
        // the trunk's lend banks: what the drafter left of each device's arena, split evenly between the layers that
        // keep experts there (a prompt takes the arena back and their residents return to the files, see
        // enter_prefill).  A group's layer takes the same count on every member, the tightest device deciding; a
        // first pass gives every layer that count, a second the leftovers one slot at a time (round robin, while every
        // member's device still has room for one)
        if (o.expert_lend && o.expert_cache) {
            std::vector<Layer*> lend_l;
            std::vector<size_t> per(devs.size(), 0);
            for (Layer& l : L) {
                if (l.mtp || !lane_file(l) || l.n_perm >= l.n_exp) continue;
                bool ok = true;
                for (int m = 0; m < l.nm; ++m) ok = ok && lane_file(member(l, m));
                if (!ok) continue;
                lend_l.push_back(&l);
                for (int m = 0; m < l.nm; ++m) per[(size_t) member(l, m).dev] += unit(l);
            }
            std::vector<size_t> left(devs.size(), 0);
            for (size_t di = 0; di < devs.size(); ++di)
                left[di] = devs[di].arena_bytes > devs[di].arena_off ? devs[di].arena_bytes - devs[di].arena_off : 0;
            std::vector<int> add(L.size(), 0);   // (by layer index: the new lend slots)
            for (Layer* lp : lend_l) {
                Layer& l = *lp;
                int n = l.n_exp - l.n_perm;
                for (int m = 0; m < l.nm; ++m) {
                    const size_t di = (size_t) member(l, m).dev;
                    n = std::min(n, per[di] > 0 ? (int) (left[di] / per[di]) : 0);
                }
                n = std::max(0, n);
                add[(size_t) l.il] = n;
                for (int m = 0; m < l.nm; ++m)   // (the even split's bytes: the round robin below spends the rest)
                    left[(size_t) member(l, m).dev] -= (size_t) n * unit(l);
            }
            for (bool grew = true; grew;) {   // (what the even split left: one slot a layer, in order)
                grew = false;
                for (Layer* lp : lend_l) {
                    Layer& l = *lp;
                    if (l.n_perm + add[(size_t) l.il] >= l.n_exp) continue;
                    bool ok = true;
                    for (int m = 0; m < l.nm && ok; ++m) {
                        const size_t di = (size_t) member(l, m).dev;
                        size_t need = 0;
                        for (int m2 = 0; m2 < l.nm; ++m2)
                            if (member(l, m2).dev == (int) di) need += unit(l);
                        ok = left[di] >= need;
                    }
                    if (!ok) continue;
                    for (int m = 0; m < l.nm; ++m) left[(size_t) member(l, m).dev] -= unit(l);
                    ++add[(size_t) l.il];
                    grew = true;
                }
            }
            for (Layer* lp : lend_l) {
                Layer& l = *lp;
                l.lend = true;
                l.n_lend = add[(size_t) l.il];
                for (int m = 0; m < l.nm; ++m) {
                    Layer& x = member(l, m);
                    Dev& d = devs[(size_t) x.dev];
                    x.lend = true;
                    x.n_lend = l.n_lend;
                    x.lend_base = d.arena + d.arena_off;
                    d.arena_off = align256(d.arena_off + (size_t) l.n_lend * unit(l));
                    d.experts += (size_t) l.n_lend * unit(l);
                }
            }
        }
        return true;
    }

    // a GGUF's VRAM experts, in file order, by a few threads through pinned staging blobs (the CPU lane's stay in the
    // file: the mapping reads them on first use)
    bool load_experts_mx(std::string& err) {
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        size_t maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < 12; ++t)
            th.emplace_back([&] {
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    bool ok = stage != nullptr && stage2 != nullptr && read_mx(l, e, stage);
                    for (int m = 0; m < l.nm && ok; ++m) {   // (a group: each member its part)
                        const Layer& x = member(l, m);
                        cudaSetDevice(devs[(size_t) x.dev].id);
                        if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                        ok = cudaMemcpy(slot_ptr(x, l.slot[(size_t) e]), l.nm > 1 ? stage2 : stage,
                                        unit(l), cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    if (!ok) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "loading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        // as load_experts in place: the VRAM experts' pages leave the cache, the most-routed of the others (the
        // profile's share of their layer, then by index) are read once, up to warm_bytes, so the cache starts with them
        struct Warm {
            int layer, e;
            double share;
        };
        std::vector<Warm> warm;
        for (const Layer& l : L) {
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            double sum = 0;
            for (int e = 0; e < l.n_exp; ++e) sum += p[e];
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0) {
                    for (int r = 0; r < 3; ++r)
                        os::advise(pk.fds[(size_t) l.mx_file[r]], (l.mx_off[r] + (uint64_t) e * l.mx_stride[r]), mx_part_bytes(l, r), false);
                } else {
                    warm.push_back({l.il, e, sum > 0 ? p[e] / sum : 0.0});
                }
            }
        }
        std::stable_sort(warm.begin(), warm.end(), [](const Warm& a, const Warm& b) { return a.share > b.share; });
        if (!ram_tier && !warm.empty()) {   // copied into RAM (copy_experts decides), the rest read ahead into what is left
            std::vector<std::pair<int, int>> hot;
            for (const Warm& w : warm) hot.push_back({w.layer, w.e});
            if (!copy_experts(hot, err)) return false;
            if (copy_mode) {
                warm.erase(std::remove_if(warm.begin(), warm.end(),
                                          [&](const Warm& w) { return L[(size_t) w.layer].ram[(size_t) w.e] >= 0; }),
                           warm.end());
                const uint64_t av = ram_avail();
                if (av != ~uint64_t{0}) warm_bytes = av > kRamMargin ? std::min<uint64_t>(warm_bytes, av - kRamMargin) : 0;
            }
        }
        if (ram_tier && warm_bytes > 0 && !warm.empty()) {
            // chunks of 32 slots up to warm_bytes (all the RAM the rest leaves, kRamMargin kept), one slot kept free for
            // a move in; the hottest experts read in, their pages then dropped from the page cache.  Plain memory, not
            // pinned: only the CPU reads it (what goes to a GPU passes the pinned staging), and pinned memory is capped on
            // Windows - the engine's own pinned buffers come after this
            const size_t want = std::min(warm.size(), (size_t) (warm_bytes / maxb));
            while (ram_slots.size() < want + 1) {
                const size_t n = std::min<size_t>(32, want + 1 - ram_slots.size());
                uint8_t* c = (uint8_t*) os::anon_alloc(n * maxb, -1, 1);
                if (c == nullptr) break;
                ram_chunks.push_back(c);
                ram_chunk_bytes.push_back(n * maxb);
                for (size_t i = 0; i < n; ++i) ram_slots.push_back(c + i * maxb);
            }
            ram_held = ram_slots.empty() ? 0 : (int) ram_slots.size() - 1;
            for (Layer& l : L)
                if (l.lay.fmt == gg::kFmtMXFP4) l.ram.assign((size_t) l.n_exp, -1);
            std::atomic<size_t> ri{0};
            std::atomic<bool> rfail{false};
            std::vector<std::thread> rt;
            for (int t = 0; t < 12 && ram_held > 0; ++t)
                rt.emplace_back([&] {
                    for (size_t i; (i = ri.fetch_add(1)) < (size_t) ram_held;) {
                        const Layer& l = L[(size_t) warm[i].layer];
                        if (!read_mx_file(l, warm[i].e, ram_slots[i])) rfail.store(true);
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) warm[i].e * l.mx_stride[r],
                                       mx_part_bytes(l, r), false);
                    }
                });
            for (auto& t : rt) t.join();
            if (rfail.load()) {
                err = "reading the RAM tier's experts from the model's files failed";
                return false;
            }
            for (int i = 0; i < ram_held; ++i) L[(size_t) warm[(size_t) i].layer].ram[(size_t) warm[(size_t) i].e] = i;
            if (ram_held > 0) ram_free.push_back(ram_held);
            if (o.expert_cache && ram_held > 0) ram_th = std::thread([this] { ram_loop(); });
            if (o.verbose)
                std::fprintf(stderr, "ds4: RAM tier: %d experts (%.1f GB of RAM), the hottest after VRAM's; the other %zu "
                                     "read from the model's files (%.1f s)\n",
                             ram_held, ram_slots.size() * (double) maxb / 1e9, warm.size() - (size_t) ram_held,
                             (now_ms() - t0) / 1000.0);
            return true;
        }
        size_t nw = 0;
        for (double b = 0; nw < warm.size() && b + (double) L[(size_t) warm[nw].layer].blob <= (double) warm_bytes; ++nw)
            b += (double) L[(size_t) warm[nw].layer].blob;
        std::atomic<size_t> wi{0};
        std::vector<std::thread> wt;
        for (int t = 0; t < 12 && nw > 0; ++t)
            wt.emplace_back([&] {
                std::vector<uint8_t> buf(maxb);
                for (size_t i; (i = wi.fetch_add(1)) < nw;) read_mx(L[(size_t) warm[i].layer], warm[i].e, buf.data());
            });
        for (auto& t : wt) t.join();
        if (o.verbose && nw > 0)
            std::fprintf(stderr, "ds4: %zu CPU-lane experts read ahead (%.1f GB) in %.1f s\n", nw,
                         nw * (double) L[0].blob / 1e9, (now_ms() - t0) / 1000.0);
        return true;
    }

    // a GGUF's CPU-lane experts copied into RAM (copy_mode), decided once VRAM's are in - on a unified-memory APU the GPU
    // has taken its share of the RAM by then, so the copies only get what it leaves.  `hot`: the CPU lane's experts, the
    // most-routed first.  --mmap-experts off: all that fit beside kRamMargin (a warning if not all do); on and auto:
    // none - read in place: the page cache follows a conversation's experts where a static set (the profile's
    // hottest) does not (the RAM tier's lesson), and copies that all fit were no faster; DS4_COPY_GB=<n>: the
    // hottest n GB (the static hybrid, for tests).  Each copy in NH parts on the NUMA nodes (DS4_COPY_NUMA=1: one), read from the files in file
    // order by a few threads, anonymous memory on huge pages (Windows: plain), not pinned (only the CPU reads it; what
    // goes to a GPU passes the pinned staging) - a layer that cannot get its memory keeps its experts in place
    bool copy_experts(const std::vector<std::pair<int, int>>& hot, std::string& err) {
        if (o.mmap_experts > 0 || hot.empty()) return true;
        const double t0 = now_ms();
        auto spare_of = [&](const Layer& l) { return o.expert_cache && l.spare_vram >= 0 ? 1 : 0; };
        double all = 0;
        {
            std::vector<char> seen(L.size(), 0);
            for (const auto& h : hot) {
                const Layer& l = L[(size_t) h.first];
                all += (double) l.blob * (1 + (seen[(size_t) l.il]++ == 0 ? spare_of(l) : 0));
            }
        }
        const uint64_t avail = ram_avail();
        // (Windows: the copies are committed memory, within what its commit still has beside the other programs' margin)
        const double can = std::min(avail == ~uint64_t{0} ? 1e300 : std::max(0.0, (double) avail - (double) kRamMargin),
                                    commit_room());
        // (auto: none - copies that all fit were measured no faster than the page cache: the CPU lane's rows are
        // compute-bound, ~46 GB/s from either on 2 Xeon sockets, NUMA parts included; a part copied cost decode speed
        // on 64 GB and on 32 GB of RAM)
        double budget = o.mmap_experts == 0 ? std::min(all, can) : 0.0;
        if (const char* v = std::getenv("DS4_COPY_GB")) budget = std::atof(v) * 1e9;
        if (o.mmap_experts == 0 && all > can)
            std::fprintf(stderr, "ds4: --mmap-experts off: the CPU lane's experts (%.1f GB) do not all fit the RAM this "
                                 "process can get (%.1f GB beside %.0f GB for the rest) - the most-routed copied, the "
                                 "others read in place\n", all / 1e9, can / 1e9, kRamMargin / 1e9);
        if (budget <= 0) {
            if (o.verbose)
                std::fprintf(stderr, "ds4: the CPU lane's experts (%.1f GB) stay in place: the RAM this process can get "
                                     "(%.1f GB beside %.0f GB for the rest) does not hold them all\n",
                             all / 1e9, can / 1e9, kRamMargin / 1e9);
            return true;
        }
        // NUMA parts: whole rows of each matrix, so any row count the nodes divide
        int nh = (int) std::min<size_t>(2, std::max<size_t>(1, cores.size()));
        if (const char* v = std::getenv("DS4_COPY_NUMA"))   // (0 / 1: one part; 2: two, even on one node - a test)
            nh = std::atoi(v) >= 2 ? 2 : 1;
        if (g.ff % nh != 0 || g.dim % nh != 0) nh = 1;
        NH = nh;
        cp_lay.assign(L.size(), {0, 0, 0, 0});
        for (const Layer& l : L) {
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                cp_lay[(size_t) l.il][(size_t) r] = off;
                off += align256((size_t) (mx_n(r) / NH) * mx_row_b(l, r));
            }
            cp_lay[(size_t) l.il][3] = off;
        }
        // the hottest that fit the budget (a layer's first one brings its spare)
        std::vector<int> n_copy(L.size(), 0);
        double used = 0;
        size_t take = 0;
        for (; take < hot.size(); ++take) {
            const Layer& l = L[(size_t) hot[take].first];
            const double add = (double) l.blob * (1 + (n_copy[(size_t) l.il] == 0 ? spare_of(l) : 0));
            if (used + add > budget) break;
            used += add;
            ++n_copy[(size_t) l.il];
        }
        // each layer's slots: one block per node of (copies + spare) parts
        std::vector<int> first(L.size(), -1);
        copy_spare.assign(L.size(), -1);
        for (Layer& l : L) {
            l.ram.assign((size_t) l.n_exp, -1);
            const int ns = n_copy[(size_t) l.il] + (n_copy[(size_t) l.il] > 0 ? spare_of(l) : 0);
            if (ns == 0) continue;
            const size_t part = cp_lay[(size_t) l.il][3], bytes = (size_t) ns * part;
            std::array<uint8_t*, 2> base = {nullptr, nullptr};
            bool ok = true;
            for (int X = 0; X < NH && ok; ++X) {
                base[(size_t) X] = (uint8_t*) node_map(bytes, X, NH);
                ok = base[(size_t) X] != nullptr;
                if (ok) {
                    ram_chunks.push_back(base[(size_t) X]);
                    ram_chunk_bytes.push_back(bytes);
                }
            }
            if (!ok) {
                std::fprintf(stderr, "ds4: no RAM for layer %d's %d copied experts (%.1f GB): they stay in place\n", l.il,
                             ns, (double) bytes * NH / 1e9);
                n_copy[(size_t) l.il] = 0;
                continue;
            }
            first[(size_t) l.il] = (int) ram_part.size();
            for (int i = 0; i < ns; ++i)
                ram_part.push_back({base[0] + (size_t) i * part, NH > 1 ? base[1] + (size_t) i * part : nullptr});
            if (spare_of(l)) copy_spare[(size_t) l.il] = first[(size_t) l.il] + ns - 1;
        }
        struct Job {
            int layer, e, slot;
        };
        std::vector<Job> jobs;
        {
            std::vector<int> next = first;
            for (size_t i = 0; i < take; ++i) {
                const int il = hot[i].first;
                if (n_copy[(size_t) il] == 0) continue;
                jobs.push_back({il, hot[i].second, next[(size_t) il]++});
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {   // (file order: the disk reads ahead)
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> ji{0};
        std::atomic<bool> bad{false};
        std::atomic<int64_t> done_b{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 12 && !jobs.empty(); ++t)
            th.emplace_back([&] {
                for (size_t i; (i = ji.fetch_add(1)) < jobs.size() && !bad.load();) {
                    const Job& j = jobs[i];
                    const Layer& l = L[(size_t) j.layer];
                    for (int r = 0; r < 3; ++r) {
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X)
                            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) j.e * l.mx_stride[r] + (uint64_t) X * nb, nb,
                                            ram_part[(size_t) j.slot][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r]))
                                bad.store(true);
                    }
                    const int64_t db = done_b.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && i % 2048 == 0)
                        std::fprintf(stderr, "ds4: copying experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
            });
        for (auto& t : th) t.join();
        if (bad.load()) {
            err = "copying the CPU lane's experts from the model's files failed";
            return false;
        }
        for (const Job& j : jobs) L[(size_t) j.layer].ram[(size_t) j.e] = j.slot;
        copy_mode = true;
        ram_held = (int) jobs.size();
        double cb = 0;
        for (const Job& j : jobs) cb += (double) L[(size_t) j.layer].blob;
        std::fprintf(stderr, "ds4: %zu of the CPU lane's %zu experts copied into RAM (%.1f GB, %d NUMA part%s) in %.1f s%s\n",
                     jobs.size(), hot.size(), cb / 1e9, NH, NH > 1 ? "s" : "", (now_ms() - t0) / 1000.0,
                     jobs.size() < hot.size() ? "; the others read in place" : "");
        return true;
    }

    bool load_experts(std::string& err) {
        if (!L.empty() && L[0].lay.fmt == gg::kFmtMXFP4) return load_experts_mx(err);
        // every expert's region, in file order, by a few threads; VRAM ones through a pinned staging blob.  In place
        // (file_experts) the CPU lane's experts are not copied: the most-routed of them (the profile's share of their
        // layer) are read once, up to warm_bytes, so the page cache starts with them, and the VRAM ones' pages are
        // dropped from it again
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        for (const Layer& l : L)
            for (int e = 0; e < l.n_exp; ++e)
                if (!file_experts || l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        uint64_t warm = 0;
        if (file_experts) {
            std::vector<std::pair<double, Job>> cand;
            for (const Layer& l : L) {
                const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) cand.push_back({sum > 0 ? p[e] / sum : 0.0, Job{l.il, e}});
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (const auto& c : cand) {
                const uint64_t b = L[(size_t) c.second.layer].region_bytes[(size_t) c.second.e];
                if (warm + b > warm_bytes) break;
                warm += b;
                jobs.push_back(c.second);
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.region_file[(size_t) a.e] != lb.region_file[(size_t) b.e])
                return la.region_file[(size_t) a.e] < lb.region_file[(size_t) b.e];
            return la.region_off[(size_t) a.e] < lb.region_off[(size_t) b.e];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        const int nth = 12;
        size_t maxr = 0, maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (uint64_t r : l.region_bytes) maxr = std::max<size_t>(maxr, r);
        }
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&] {
                std::vector<uint8_t> region(maxr);
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                int cur_dev = -1;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    if (!pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], l.region_bytes[(size_t) e],
                                    region.data())) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "reading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    if (l.slot[(size_t) e] >= 0) {
                        repack(l, region.data(), stage);
                        bool up = true;
                        for (int m = 0; m < l.nm && up; ++m) {   // (a group: each member its part)
                            const Layer& x = member(l, m);
                            const Dev& d = devs[(size_t) x.dev];
                            if (cur_dev != d.id) {
                                cudaSetDevice(d.id);
                                cur_dev = d.id;
                            }
                            if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                            up = cudaMemcpy(slot_ptr(x, l.slot[(size_t) e]), l.nm > 1 ? stage2 : stage,
                                            unit(x), cudaMemcpyHostToDevice) == cudaSuccess;
                        }
                        if (!up) {
                            std::lock_guard<std::mutex> lk(emu);
                            ferr = "uploading layer " + std::to_string(l.il) + " expert " + std::to_string(e);
                            failed.store(true);
                            break;
                        }
                        if (file_experts)
                            os::advise(pk.fds[(size_t) l.region_file[(size_t) e]], l.region_off[(size_t) e], l.region_bytes[(size_t) e], false);
                    } else if (!file_experts) {
                        for (int X = 0; X < NH; ++X) repack_part(l, region.data(), X, l.hh[(size_t) e][(size_t) X]);
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.region_bytes[(size_t) e]) +
                                       (int64_t) l.region_bytes[(size_t) e];
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(),
                                     db / 1e9, db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        if (file_experts)
            std::fprintf(stderr, "ds4: experts loaded in %.1f s (%.1f GB of the CPU lane's read ahead)\n",
                         (now_ms() - t0) / 1000.0, warm / 1e9);
        return true;
    }

    std::string describe() const {
        std::ostringstream s;
        for (size_t di = 0; di < devs.size(); ++di) {
            const Dev& d = devs[di];
            int a = -1, b = -1, ev = 0, el = 0, et = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di) {
                        if (a < 0) a = l.il;
                        b = l.il;
                        ev += l.n_vram;
                        el += l.mtp ? 0 : l.n_lend;
                        et += l.n_exp;
                    }
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            const size_t gsz = groups.empty() ? 1 : groups[(size_t) d.grp].mem.size();
            char tp[64] = "";
            if (gsz > 1) std::snprintf(tp, sizeof tp, " (tensor parallel %zu, part %d)", gsz, d.mi);
            char ar[40] = "", ln[32] = "";
            if (d.arena_bytes > 0) std::snprintf(ar, sizeof ar, "%.2f GB", d.arena_bytes / 1e9);
            else std::snprintf(ar, sizeof ar, "none");
            if (el > 0) std::snprintf(ln, sizeof ln, " + %d", el);
            char line[448];
            std::snprintf(line, sizeof line,
                          "ds4: CUDA%d layers [%d, %d]%s%s: weights %.2f GB, experts %.2f GB (%d of %d in VRAM%s, %d "
                          "lent while decoding), caches %.2f GB, arena %s, %.0f MiB free\n",
                          d.id, a, b, tp, (int) di == head_dev ? " + head" : "", d.weights / 1e9, d.experts / 1e9, ev, et,
                          ln, el, d.caches / 1e9, ar, fr / 1048576.0);
            s << line;
        }
        int nh = 0;
        size_t hb_ = 0;
        for (const Layer& l : L) {
            nh += l.n_exp - (l.mtp ? l.n_vram : l.n_vram + l.n_lend);   // (the decode layout: perm + lent)
            hb_ += (size_t) (l.n_exp - l.n_vram) * l.blob;
        }
        char line[200];
        if (copy_mode)
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB): %d copied into RAM (%d NUMA "
                                             "parts), the rest in place in the model's files; %d threads\n",
                          nh, hb_ / 1e9, ram_held, NH, cpu_threads());
        else if (file_experts)
            std::snprintf(line, sizeof line,
                          "ds4: %d routed experts on the CPU lane (%.1f GB, in place in the model's files), %d threads\n",
                          nh, hb_ / 1e9, cpu_threads());
        else
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB of RAM, %d NUMA parts), %d threads\n", nh,
                          hb_ / 1e9, NH, cpu_threads());
        s << line;
        return s.str();
    }

    int cpu_threads() const {
        int n = 0;
        for (const auto& pl : pools) n += pl->size() - 1;
        return n + 1;
    }
    // the CPU lane's workers: a pool per NUMA node (NH of them) on its cores; in place with one part, one pool over
    // every node's cores (the pages lie where the cache put them).  The physical cores pin first, exactly as before;
    // the logical siblings follow, round-robin over the nodes, so --cpu-threads past the physical cores puts the
    // extra workers on SMT threads instead of clipping the pool
    void make_pools() {
        std::vector<std::vector<int>> cs = cores, as = all_cpus;
        if (as.size() < cs.size()) as.resize(cs.size());
        if ((int) cs.size() < NH && !cs.empty()) {   // (DS4_COPY_NUMA=2 on one node, a test: its cores split in NH)
            std::vector<int> all = cs[0];
            cs.assign((size_t) NH, {});
            for (size_t i = 0; i < all.size(); ++i) cs[i * (size_t) NH / all.size()].push_back(all[i]);
            as = cs;   // (the test's split has no sibling lists)
        }
        for (int X = 0; X < (int) cs.size() && X < std::max(NH, 1); ++X) {
            std::vector<int> c = cs[(size_t) X];
            if (file_experts && NH == 1)
                for (size_t Y = 1; Y < cs.size(); ++Y) c.insert(c.end(), cs[Y].begin(), cs[Y].end());
            const int host_core = X == 0 && !c.empty() ? c.front() : -1;
            if (X == 0 && !c.empty()) c.erase(c.begin());   // the host thread's core (it helps node 0's pool)
            std::vector<int> list = c;
            auto add = [&](int q) {
                if (q != host_core && std::find(list.begin(), list.end(), q) == list.end()) list.push_back(q);
            };
            if (file_experts && NH == 1) {   // (the one pool: the nodes' siblings alternate)
                for (size_t i = 0;; ++i) {
                    bool more = false;
                    for (size_t Y = 0; Y < as.size(); ++Y)
                        if (i < as[Y].size()) {
                            add(as[Y][i]);
                            more = true;
                        }
                    if (!more) break;
                }
            } else if ((size_t) X < as.size()) {
                for (int q : as[(size_t) X]) add(q);
            }
            int nw = (int) c.size();
            if (o.cpu_threads > 0)
                nw = std::max(X == 0 ? 0 : 1, std::min((int) list.size(), o.cpu_threads / NH - (X == 0 ? 1 : 0)));
            pools.push_back(std::make_unique<Pool>(nw, list, 20000));   // spin through a token's layers (20 ms)
        }
        (void) gg::mx_tier();   // (measured here, not on the first token)
        if (o.verbose) {
            int nthr = 0;
            for (auto& p : pools) nthr += p->size();
            std::fprintf(stderr, "ds4: CPU lane: %d threads, MXFP4 rows %s (%s)\n", nthr, gg::mx_tier_name(gg::mx_tier()),
                         gg::mx_tier_report());
        }
    }

    void reset() {
        pos = 0;
        toks.clear();
        ctoks.clear();
        ck[0].pos = ck[1].pos = -1;
        for (Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                Layer& l = member(l0, m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemsetAsync(l.win, 0, (size_t) g.win * win_rb(), d.st);
                if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
                if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) {
        Check& c = ck[slot & 1];
        c.pos = pos;
        c.win.resize(L.size());
        c.comp.resize(L.size() * 2);
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            c.win[i].resize((size_t) g.win * win_rb());
            cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size(), cudaMemcpyDeviceToHost, d.st);
            if (l.c_kv) {
                c.comp[2 * i].resize((size_t) l.ratio * g.hd);
                c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
                cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyDeviceToHost, d.st);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the checkpoint at or before position limit (the state's tokens agree with a new prompt up to there), or -1
    int best_check(int64_t limit) const {
        int b = -1;
        for (int i = 0; i < 2; ++i)
            if (ck[i].pos >= 0 && ck[i].pos <= limit && ck[i].pos <= pos && (b < 0 || ck[i].pos > ck[b].pos)) b = i;
        return b;
    }
    int64_t restorable(int64_t limit) const {
        const int b = best_check(limit);
        return b < 0 ? -1 : ck[b].pos;
    }

    bool restore(int64_t limit) {
        const int b = best_check(limit);
        if (b < 0) return false;
        const Check& c = ck[b];
        for (size_t i = 0; i < L.size(); ++i)
            for (int m = 0; m < L[i].nm; ++m) {   // (a group's members hold the same state: the lead's copy)
                Layer& l = member(L[i], m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size(), cudaMemcpyHostToDevice, d.st);
                if (l.c_kv) {
                    cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                    cudaMemcpyHostToDevice, d.st);
                }
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        pos = c.pos;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
            if (o2.pos > pos) o2.pos = -1;
        return true;
    }

    size_t bounce_bytes() const {
        return (size_t) kWin * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    size_t topk_stride() const { return (size_t) std::max(g.ix_topk, 32); }
    size_t cand_stride() const { return (size_t) std::max(g.cand_topk, 32); }
    size_t bm_stride() const { return (size_t) (o.max_context + 64) / std::max(1, g.cand_blk) + 64; }
    static constexpr int kSlots = kWin * kPicks;   // the CPU lane's (token, expert) rows a window can need

    // ---- the vision tower (the checkpoint's vision.* / aligner.*) on the first GPU, its weights streamed from the
    // host; a prompt's pictures as their spans' rows (set_images)
    VisionTower vt;
    VisionGeo vg;
    int img_tok = -1;
    std::string vision_err;              // why the tower did not load (pictures off), or empty
    std::vector<uint16_t> img_mark[3];   // image_start, image_newline, image_end: BF16 [dim]
    std::vector<Image> imgs;             // the pictures of the prompt being fed, by start
    bool load_vision(std::string& err) {
        vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
        const int tok = (int) pk.num("image_token_id", -1);
        if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
        vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
        vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
        vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
        vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
        const char* mark[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const Tensor* t = pk.get(mark[i]);
            img_mark[i].resize((size_t) g.dim);
            if (t == nullptr || t->dtype != "BF16" || t->bytes != (uint64_t) g.dim * 2 || !pk.read(*t, img_mark[i].data())) {
                err = std::string(mark[i]) + ": not a BF16 row of the model's width";
                return false;
            }
        }
        const double t0 = now_ms();
        Dev& d = devs[0];
        auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
            const Tensor* t = pk.get(n);
            if (t == nullptr || t->dtype != "BF16") return false;
            shape = t->shape;
            return dst == nullptr || pk.read(*t, dst);
        };
        if (!vt.load(vg, d.id, read, err)) return false;
        img_tok = tok;
        CK(cudaSetDevice(d.id));
        CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
        if (o.verbose)
            std::fprintf(stderr, "ds4: vision tower (%d blocks) on CUDA%d: %.0f MiB, up to %d patches, %.1f s\n",
                         vg.layers, d.id, (double) vt.device_bytes() / (1 << 20), vt.max_patches(),
                         (now_ms() - t0) / 1000.0);
        return true;
    }
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
        if (img_tok < 0) {
            err = "this model has no vision tower (or it is off)";
            return false;
        }
        std::vector<uint16_t> rows;
        int nh = 0, nw = 0;
        if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
        const size_t D = (size_t) g.dim;
        span.resize(((size_t) nh * (nw + 1) + 2) * D);
        uint16_t* o = span.data();
        auto put = [&](const uint16_t* r) {
            std::memcpy(o, r, D * 2);
            o += D;
        };
        put(img_mark[0].data());
        for (int i = 0; i < nh; ++i) {
            for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
            put(img_mark[1].data());
        }
        put(img_mark[2].data());
        return true;
    }
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const {
        if (imgs.empty()) return nullptr;
        auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
        if (it == imgs.begin()) return nullptr;
        const Image& im = *(it - 1);
        const int64_t r = p - im.start;
        if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
        return im.rows->data() + (size_t) r * g.dim;
    }
    // token tk's embedding row at position p: inside a picture's span, the picture's
    const uint16_t* emb_row(int64_t p, int tk) const {
        const uint16_t* r = tk == img_tok ? image_row(p) : nullptr;
        return r != nullptr ? r : embed_pinned + (size_t) tk * g.dim;
    }
    bool in_image(int64_t p) const { return img_tok >= 0 && toks[(size_t) p] == img_tok && image_row(p) != nullptr; }
    // the Engram's compressed id of token tk at p (-1: a picture's, which no n-gram spans)
    int32_t ctok(int64_t p, int tk) const {
        if (tk == img_tok && image_row(p) != nullptr) return -1;
        return tk < pk.eg_vocab ? pk.eg_map[(size_t) tk] : 0;
    }
    // a chunk's image rows (route()'s bits); false: none
    bool vl_bits(int64_t p0, int T, VlBits& b) const {
        if (imgs.empty()) return false;
        std::memset(&b, 0, sizeof b);
        bool any = false;
        for (int t = 0; t < T && t < kVlRows; ++t)
            if (in_image(p0 + t)) {
                b.w[t >> 5] |= 1u << (t & 31);
                any = true;
            }
        return any;
    }

    // ---- Engram: a window's rows of one table (Engram layer li), tokens p0 .. p0 + T - 1 into dst ([token] of [cols]
    // [hd] codes then the scales; nullptr: h_eg).  eg_issue serves the rows already fetched from the row cache (and a
    // row the window needs twice from one read) and hands the rest to the reader's threads; eg_finish waits for them,
    // copies the repeats and keeps the new rows in the cache.  The rows are the file's bytes either way.
    // the Engram table's row format (ds4_kernels.cuh's engram_dequant fmt): 0 MXFP8 codes + scales, 1 Q6_K rows
    int eg_fmt(int li) const {
        const Tensor* tw = pk.get("layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.weight");
        return tw != nullptr && tw->dtype == "Q6_K" ? 1 : 0;
    }
    void eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, hd = pk.eg_hd;
        const size_t per = (size_t) cols * (hd + hd / 32);
        b.clear();
        b.reads.reserve((size_t) T * cols);
        const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
        const Tensor* tw = pk.get(P + "weight");
        const Tensor* ts = pk.get(P + "scale");
        const bool mx8 = tw->dtype == "MXFP8";   // (a GGUF: the scales between the codes, one per 32)
        const bool q6 = tw->dtype == "Q6_K";     // (one block a row: its 16 scales inside it)
        const size_t rb = q6 ? (size_t) (hd / 256) * 210 : (size_t) hd;   // a column's payload bytes in its slot
        const size_t sb = q6 ? 0 : (size_t) (hd / 32);                    // ... and its scale bytes after them
        std::unordered_map<uint64_t, int> first;   // a row this batch reads already -> its read
        int64_t hits = 0;
        {
            std::lock_guard<std::mutex> lk(eg_cache_mu);
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                uint8_t* rows = dst != nullptr ? dst + (size_t) t * per : h_eg + ((size_t) li * eg_cap + t) * per;
                uint8_t* scales = rows + (size_t) cols * hd;
                if (ctoks[(size_t) p] < 0) {   // a picture's token: no rows (zeros: its gate adds nothing)
                    std::memset(rows, 0, per);
                    continue;
                }
                int64_t prod[8];
                bool blocked = false;
                for (int s = 0; s < pk.eg_ngram; ++s) {   // (the look-back stops at the start and at a picture)
                    blocked = blocked || p < s || ctoks[(size_t) (p - s)] < 0;
                    const int64_t tk = blocked ? pk.eg_pad : ctoks[(size_t) (p - s)];
                    prod[s] = tk * pk.eg_mult[(size_t) li * pk.eg_ngram + s];
                }
                int64_t rolling = prod[0];
                for (int n = 1; n < pk.eg_ngram; ++n) {
                    rolling ^= prod[n];
                    for (int h = 0; h < pk.eg_heads; ++h) {
                        const int c = (n - 1) * pk.eg_heads + h;
                        const int64_t id = rolling % pk.eg_primes[(size_t) li * cols + c] + pk.eg_off[(size_t) li * cols + c];
                        uint8_t* rd = rows + (size_t) c * hd;
                        uint8_t* sd = scales + (size_t) c * (hd / 32);
                        const uint64_t key = (uint64_t) id * 8 + (uint64_t) li;
                        if (const uint8_t* hit = eg_cache.find(key)) {
                            std::memcpy(rd, hit, rb);
                            if (sb != 0) std::memcpy(sd, hit + hd, sb);
                            ++hits;
                            continue;
                        }
                        const auto fi = first.find(key);
                        if (fi != first.end()) {
                            b.dups.push_back({fi->second, {rd, sd}});
                            continue;
                        }
                        first.emplace(key, (int) b.reads.size());
                        EgRead r;
                        r.file = tw->file;
                        r.dst = rd;
                        r.scales = sd;
                        r.key = key;
                        if (mx8) {
                            r.mx8 = true;
                            r.n = (uint32_t) (hd / 32 * 33);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else if (q6) {
                            r.n = (uint32_t) rb;
                            r.off = tw->off + (uint64_t) id * rb;
                            r.scales = nullptr;
                        } else {
                            r.n = (uint32_t) hd;
                            r.off = tw->off + (uint64_t) id * hd;
                            r.file2 = ts->file;
                            r.n2 = (uint32_t) (hd / 32);
                            r.off2 = ts->off + (uint64_t) id * (hd / 32);
                        }
                        b.reads.push_back(r);
                    }
                }
            }
        }
        eg_rows_n += hits + (int64_t) b.reads.size() + (int64_t) b.dups.size();
        eg_hits_n += hits + (int64_t) b.dups.size();
        eg_reads_n += (int64_t) b.reads.size();
        eg_io.submit(b);
    }
    // b's rows all in place (any thread; a group's members may ask at once: one finishes, the others wait for it)
    void eg_finish(EgBatch& b) {
        if (b.post.load(std::memory_order_acquire) == 2) return;
        const int64_t t0 = now_ns();
        eg_io.wait(b);
        int want = 0;
        if (b.post.compare_exchange_strong(want, 1, std::memory_order_acq_rel)) {
            const int hd = pk.eg_hd;
            // (r.n / r.n2 are the read's sizes: an mx8 row's packed bytes deinterleave to hd codes and hd/32 scales)
            auto n_codes = [&](const EgRead& r) { return r.mx8 ? (size_t) hd : (size_t) r.n; };
            auto n_scales = [&](const EgRead& r) { return r.mx8 ? (size_t) (hd / 32) : (size_t) r.n2; };
            for (const auto& d : b.dups) {
                const EgRead& r = b.reads[(size_t) d.first];
                std::memcpy(d.second.first, r.dst, n_codes(r));
                if (n_scales(r) != 0) std::memcpy(d.second.second, r.scales, n_scales(r));
            }
            {
                std::lock_guard<std::mutex> lk(eg_cache_mu);
                for (size_t i = 0; i < b.reads.size(); ++i) {
                    const EgRead& r = b.reads[i];
                    if (!b.ok[i]) {   // (a failed read: zeros, its gate adds nothing; not kept)
                        std::memset(r.dst, 0, n_codes(r));
                        if (n_scales(r) != 0) std::memset(r.scales, 0, n_scales(r));
                        continue;
                    }
                    if (uint8_t* c = eg_cache.put(r.key)) {
                        std::memcpy(c, r.dst, n_codes(r));
                        if (n_scales(r) != 0) std::memcpy(c + hd, r.scales, n_scales(r));
                    }
                }
            }
            eg_io_ns += b.t1.load() - b.t0;
            b.live = false;
            b.post.store(2, std::memory_order_release);
        } else {
            while (b.post.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        }
        eg_wait_ns += now_ns() - t0;
    }
    void eg_stats() {
        st.eg_rows = eg_rows_n.load();
        st.eg_hits = eg_hits_n.load();
        st.eg_reads = eg_reads_n.load();
        st.eg_io_ms = (double) eg_io_ns.load() / 1e6;
        std::lock_guard<std::mutex> lk(eg_cache_mu);
        st.eg_cached = (int64_t) eg_cache.size();
        st.eg_cache_cap = (int64_t) eg_cache.capacity();
    }
    // the chunked prompt path: rows of Engram layer `only` into dst, now (a helper thread, a chunk ahead)
    void engram_rows(int64_t p0, int T, int only, uint8_t* dst) {
        EgBatch b;
        eg_issue(p0, T, only, dst, b);
        eg_finish(b);
    }

    // ---- the CPU lane: a window's routed experts that live on the host, as entries (an expert and up to 8 of its
    // tokens, in rows slot0 ..) computed together: each node its column part of every matrix, from memory on that node,
    // in four rounds (the gate / up split-K pieces, each preparing its two input blocks of every token; each band's
    // sums / codebook / Hadamard / swiglu and that block of the down input, which both nodes then read whole; the down
    // pieces; the down bands).  yd[slot] gets each row's expert output.
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
        int tok[kc::kExl3CpuMaxN] = {};
    };
    void run_nodes(int n, const std::function<void(int)>* f) {
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }

    // a GGUF's MXFP4 experts on the CPU lane, read from the file's mapping or their copy in RAM (mx_part_ptr): each
    // node its rows of every matrix - the gate / up rows [X n_ff / NH, ..) of each entry (the swiglu into its h row),
    // then (h whole) the down rows [X dim / NH, ..) into yd.  A row's sum is the same whatever the split: the same bits
    // in a copy as in place.  The device's arithmetic (gg::mx_rows: int8 activations rounded as the device rounds
    // them, exact integer sums a block; every CPU tier the same bits): each token's input quantized once, whatever
    // number of entries take it, and each job its band of h straight into the down's int8 input.
    // (ord: the order the entries are computed in - their slots, so the sums, stay the same; null: as they are)
    void cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
        if (!gg::mx_experts(l.lay)) return cpu_experts_iq(l, x, ents, ldx, ord);
        constexpr int RB = 64;   // rows a job (whole blocks of h: RB % 32 == 0)
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        const size_t gr = gg::mx_row_bytes(E), dr = gg::mx_row_bytes(FF);
        const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
        const bool band_q = FFh % 32 == 0;   // (each job's rows whole blocks of h)
        const float lim = g.lim;
        static_assert(kc::kExl3CpuMaxN <= gg::kMxCpuMaxN, "a CPU-lane entry's tokens in one call");
        {   // the experts read from the files (not in the RAM tier) that are not resident: all their pages asked for at
            // once, so the SSD reads them with large requests side by side while the first entries compute
            // (os::prefetch_cold: the resident ones are left alone - the hint walks every page of a range, which cost
            // more than computing a warm expert, 0.34 ms a 6.3 MB matrix on Windows)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (l.ram.empty() || l.ram[(size_t) en.e] < 0)
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            os::prefetch_cold(want);
        }
        // the int8 inputs in c_xh's bytes: token t's (t < kWin) at t xa, then the slots' h at (kWin + slot) ha; a token
        // index past the window (none on the decode or chunk paths) gets a row of its own after them
        uint8_t* xq = (uint8_t*) c_xh.data();
        uint8_t* hq = xq + (size_t) kWin * xa;
        uint8_t* xq_extra = hq + (size_t) kSlots * ha;
        auto x_act = [&](const Ent& en, int i) -> const void* {
            const int t = en.tok[i];
            return t < kWin ? xq + (size_t) t * xa : xq_extra + (size_t) (en.slot0 + i) * xa;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::mx_quant(x + (size_t) t * ldx, E, (void*) x_act(en, i));
                }
        }
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float yg[gg::kMxCpuMaxN][RB], yu[gg::kMxCpuMaxN][RB];
                float *gp[gg::kMxCpuMaxN], *up[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    gp[i] = yg[i];
                    up[i] = yu[i];
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's)
                gg::mx_rows(mx_part_ptr(l, en.e, 0, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, gp);
                gg::mx_rows(mx_part_ptr(l, en.e, 1, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, up);
                for (int i = 0; i < en.n; ++i) {
                    float* h = band_q ? yg[i] : c_xhd.data() + (size_t) (en.slot0 + i) * FF + r0;
                    for (int r = r0; r < r1; ++r) {
                        float gv = yg[i][r - r0], uv = yu[i][r - r0];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        h[r - r0] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    if (band_q) gg::mx_quant_blocks(h, r0 / 32, r1 / 32, hq + (size_t) (en.slot0 + i) * ha, FF);
                }
            };
        run_nodes(ne * GB, f);
        if (!band_q)
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i)
                    gg::mx_quant(c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF, hq + (size_t) (en.slot0 + i) * ha);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float* yp[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = hq + (size_t) (en.slot0 + i) * ha;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + r0;
                }
                gg::mx_rows(mx_part_ptr(l, en.e, 2, X), dr, FF, r0 - X * Eh, r1 - X * Eh, ap, en.n, yp);
            };
        run_nodes(ne * DB, f);
    }
    // ... the low-bit ones (ds4_iq's types): as above, through ggml-cpu's dots (gg::iq_cpu_*) - each token's input
    // quantized once to the weight type's activation (Q8_K) in its c_xh row's bytes, then (h whole) its h the same
    void cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
        constexpr int RB = 64;   // rows a job
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        gg::IqCpu fm;
        std::string ferr;
        if (!gg::iq_cpu_fmt(gg::expert_type(l.lay, 0), gg::expert_type(l.lay, 2), E, FF, fm, ferr) ||
            fm.act_bytes > (size_t) E * 4 || fm.h_bytes > (size_t) E * 4) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) std::fprintf(stderr, "ds4: layer %d's CPU-lane experts: %s\n", l.il, ferr.c_str());
            for (const Ent& en : ents) std::memset(c_yd.data() + (size_t) en.slot0 * E, 0, (size_t) en.n * E * 4);
            return;
        }
        const float lim = g.lim;
        {   // (os::prefetch, as above)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (l.ram.empty() || l.ram[(size_t) en.e] < 0)
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            os::prefetch(want);
        }
        // the activations in c_xh's bytes, E 4 apart: token t's (t < kWin) at row t - quantized once, whatever number of
        // entries take it - then a row a slot (kWin + slot): a token past the window's input (none on the decode or
        // chunk paths), then (x no longer needed) the slot's h
        uint8_t* aq = (uint8_t*) c_xh.data();
        auto x_act = [&](const Ent& en, int i) -> uint8_t* {
            const int t = en.tok[i];
            return aq + (size_t) (t < kWin ? t : kWin + en.slot0 + i) * E * 4;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * ldx, E, x_act(en, i));
                }
        }
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float* hp[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    hp[i] = c_xhd.data() + (size_t) (en.slot0 + i) * FF + (size_t) X * FFh;   // (part X's rows of h)
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's - whole rows, so any
                // block size of the type)
                gg::iq_cpu_gate_up(fm, mx_part_ptr(l, en.e, 0, X), mx_part_ptr(l, en.e, 1, X), ap, en.n, hp,
                                   r0 - X * FFh, r1 - X * FFh, lim);
            };
        run_nodes(ne * GB, f);
        for (const Ent& en : ents)
            for (int i = 0; i < en.n; ++i)
                gg::iq_cpu_quant(fm.d_act, c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF,
                                 aq + (size_t) (kWin + en.slot0 + i) * E * 4);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* hp[gg::kMxCpuMaxN];
                float* yp[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    hp[i] = aq + (size_t) (kWin + en.slot0 + i) * E * 4;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + (size_t) X * Eh;   // (part X's rows of it)
                }
                gg::iq_cpu_down(fm, mx_part_ptr(l, en.e, 2, X), hp, en.n, yp, r0 - X * Eh, r1 - X * Eh);
            };
        run_nodes(ne * DB, f);
    }

    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx = 0,
                     const int* ord = nullptr) {
        if (ldx == 0) ldx = (size_t) g.dim;
        if (l.lay.fmt == gg::kFmtMXFP4) return cpu_experts_mx(l, x, ents, ldx, ord);
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
        const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
        const int ne = (int) ents.size();
        const float lim = g.lim;
        auto part = [&](int e, int X, int r) {
            const uint8_t* b = l.hh[(size_t) e][(size_t) X];
            kc::Exl3Part p;
            p.trellis = (const uint32_t*) (b + l.hl.trellis[r]);
            p.suh = (const uint16_t*) (b + l.hl.suh[r]);
            p.svh = (const uint16_t*) (b + l.hl.svh[r]);
            p.k = l.hl.k[r];
            p.n = l.hl.n[r];
            p.K = l.lay.K[r];
            return p;
        };
        auto xh = [&](int X, int sl, int r) { return c_xh.data() + ((size_t) (X * kSlots + sl) * 2 + r) * E; };
        auto xsb = [&](int X, int sl, int r) { return c_xsb.data() + ((size_t) (X * kSlots + sl) * 2 + r) * EB; };
        auto acc = [&](int X, int sl, int r, int q) {
            return c_acc.data() + (((size_t) (X * kSlots + sl) * 2 + r) * GQ + q) * FFh;
        };
        auto xhd = [&](int sl) { return c_xhd.data() + (size_t) sl * FF; };
        auto xsd = [&](int sl) { return c_xsd.data() + (size_t) sl * FB; };
        auto accd = [&](int X, int sl, int q) { return c_accd.data() + ((size_t) (X * kSlots + sl) * DQ + q) * Eh; };
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / (2 * GQ))];
                const int r = (j / GQ) % 2, q = j % GQ;
                const kc::Exl3Part p = part(en.e, X, r);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    const float* xt = x + (size_t) en.tok[i] * ldx;
                    for (int b = 2 * q; b < 2 * q + 2; ++b)
                        xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
                    xp[i] = xh(X, sl, r);
                    ap[i] = acc(X, sl, r, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
            };
        run_nodes(ne * 2 * GQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / GB)];
                const int b = j % GB;
                const kc::Exl3Part pg = part(en.e, X, 0), pu = part(en.e, X, 1), pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float y[2][128], a[128], hb[128];
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                        for (int q = 0; q < GQ; ++q) {
                            const float* aq = acc(X, sl, r, q) + b * 128;
                            for (int c = 0; c < 128; ++c) a[c] += aq[c];
                        }
                        double xs = 0.0;
                        for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                        kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
                    }
                    for (int c = 0; c < 128; ++c) {
                        float gv = y[0][c], uv = y[1][c];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        hb[c] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    const int gb = X * GB + b;
                    xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
                }
            };
        run_nodes(ne * GB, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DQ)];
                const int q = j % DQ;
                const kc::Exl3Part p = part(en.e, X, 2);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    xp[i] = xhd(en.slot0 + i);
                    ap[i] = accd(X, en.slot0 + i, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
            };
        run_nodes(ne * DQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DB)];
                const int b = j % DB;
                const kc::Exl3Part pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float a[128];
                    for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                    for (int q = 0; q < DQ; ++q) {
                        const float* aq = accd(X, sl, q) + b * 128;
                        for (int c = 0; c < 128; ++c) a[c] += aq[c];
                    }
                    double xs = 0.0;
                    for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
                    kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c_yd.data() + (size_t) sl * E + (size_t) X * Eh + b * 128);
                }
            };
        run_nodes(ne * DB, f);
    }

    // ---- the expert cache.  A swap moves a CPU-lane expert (`in`) into the layer's spare VRAM slot and the VRAM expert it
    // replaces (`out`) into the spare host parts - both copies of weights that do not change, made while the model runs
    // (the copy stream, the host's repacking) - and cache_tick switches the tables between windows, so the device's
    // table and the host's view of a layer always agree within a window.  In place (file_experts) `out` needs no copy:
    // the CPU lane reads it from the file again.
    void ram_loop() {   // the RAM tier's moves in: an expert read from the file into a free slot
        for (;;) {
            RamSwap rs;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                ram_cv.wait(lk, [&] { return quit.load() || !ram_queue.empty(); });
                if (quit.load()) return;
                rs = ram_queue.front();
                ram_queue.pop_front();
            }
            const Layer& l = L[(size_t) rs.layer];
            if (read_mx_file(l, rs.in, ram_slots[(size_t) rs.slot])) {
                for (int r = 0; r < 3; ++r)   // (one copy of it in RAM)
                    os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) rs.in * l.mx_stride[r],
                               mx_part_bytes(l, r), false);
            } else {
                rs.in = -1;   // (failed: the slot goes back)
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_ready.push_back(rs);
        }
    }

    void cache_loop(SwapBufs& B) {
        for (;;) {
            Swap sw;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                sw_cv.wait(lk, [&] { return quit.load() || !sw_queue.empty(); });
                if (quit.load()) return;
                sw = sw_queue.front();
                sw_queue.pop_front();
            }
            Layer& l = L[(size_t) sw.layer];
            if (l.lay.fmt == gg::kFmtMXFP4) {   // in: from the file; out: nothing to keep (the file has it)
                swap_in_mx(l, sw, B);
                std::lock_guard<std::mutex> lk(sw_mu);
                sw_ready.push_back(sw);
                continue;
            }
            parts_to_blob(l, l.hh[(size_t) sw.in], B.a);
            if (l.nm == 1) {
                const Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaStream_t cs = B.st[(size_t) l.dev];
                cudaMemcpyAsync(slot_ptr(l, l.spare_vram), B.a, l.blob, cudaMemcpyHostToDevice, cs);
                if (!file_experts)
                    cudaMemcpyAsync(B.b, slot_ptr(l, l.slot[(size_t) sw.out]), l.blob,
                                    cudaMemcpyDeviceToHost, cs);
                cudaStreamSynchronize(cs);
            } else {   // a group: each member's part in, each member's part out
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    cudaSetDevice(devs[(size_t) x.dev].id);
                    cudaStream_t cs = B.st[(size_t) x.dev];
                    blob_to_half(l, B.a, m, B.h);
                    cudaMemcpyAsync(slot_ptr(x, l.spare_vram), B.h, x.hblob, cudaMemcpyHostToDevice, cs);
                    cudaStreamSynchronize(cs);
                    if (file_experts) continue;
                    cudaMemcpyAsync(B.h, slot_ptr(x, l.slot[(size_t) sw.out]), x.hblob,
                                    cudaMemcpyDeviceToHost, cs);
                    cudaStreamSynchronize(cs);
                    half_to_blob(l, B.h, m, B.b);
                }
            }
            if (!file_experts)
                for (int X = 0; X < NH; ++X) blob_to_part(l, B.b, X, l.spare_host[(size_t) X]);
            if (check_swaps) {   // DS4_CHECK_SWAPS: both moves against the model file, byte for byte
                std::vector<uint8_t> reg, blob(l.blob), part(l.hl.bytes);
                bool ok = true;
                for (int pass = 0; pass < (file_experts ? 1 : 2); ++pass) {   // (in place: no host copy of `out`)
                    const int e = pass == 0 ? sw.in : sw.out;
                    reg.resize(l.region_bytes[(size_t) e]);
                    pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], reg.size(), reg.data());
                    if (pass == 0) {
                        repack(l, reg.data(), blob.data());
                        std::vector<uint8_t> dev(l.blob);
                        if (l.nm == 1)
                            cudaMemcpy(dev.data(), slot_ptr(l, l.spare_vram), l.blob, cudaMemcpyDeviceToHost);
                        else
                            for (int m = 0; m < l.nm; ++m) {   // (the members' parts, put together again)
                                std::vector<uint8_t> h(member(l, m).hblob);
                                cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                                cudaMemcpy(h.data(), slot_ptr(member(l, m), l.spare_vram), member(l, m).hblob,
                                           cudaMemcpyDeviceToHost);
                                half_to_blob(l, h.data(), m, dev.data());
                            }
                        for (int r = 0; r < 3; ++r)
                            ok = ok && std::memcmp(dev.data() + l.lay.trellis[r], blob.data() + l.lay.trellis[r], l.piece_bytes[r][0]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.suh[r], blob.data() + l.lay.suh[r], l.piece_bytes[r][1]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.svh[r], blob.data() + l.lay.svh[r], l.piece_bytes[r][2]) == 0;
                    } else {
                        for (int X = 0; X < NH; ++X) {
                            repack_part(l, reg.data(), X, part.data());
                            for (int r = 0; r < 3; ++r) {
                                const size_t tb = (size_t) (l.hl.k[r] / 16) * (l.hl.n[r] / 16) * 32 * l.lay.K[r];
                                ok = ok && std::memcmp(part.data() + l.hl.trellis[r], l.spare_host[(size_t) X] + l.hl.trellis[r], tb) == 0 &&
                                     std::memcmp(part.data() + l.hl.suh[r], l.spare_host[(size_t) X] + l.hl.suh[r], (size_t) l.hl.k[r] * 2) == 0 &&
                                     std::memcmp(part.data() + l.hl.svh[r], l.spare_host[(size_t) X] + l.hl.svh[r], (size_t) l.hl.n[r] * 2) == 0;
                            }
                        }
                    }
                }
                std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out: %s\n", l.il, sw.in, sw.out,
                             ok ? "both match the model file" : "MISMATCH");
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            sw_ready.push_back(sw);
        }
    }

    // a GGUF's swap: expert sw.in from the file (copy_mode: its copy) into the spare VRAM slot (each member its part);
    // the expert it replaces goes back to the CPU lane, which reads it from the file's mapping - or (copy_mode, sw.ram)
    // from the layer's spare RAM slot, which it is copied into from VRAM here
    void swap_in_mx(Layer& l, Swap& sw, SwapBufs& B) {
        uint8_t *sw_a = B.a, *sw_b = B.b, *sw_h = B.h;   // (this thread's)
        const std::vector<cudaStream_t>& copy_st = B.st;
        bool ok = copy_mode ? read_mx(l, sw.in, sw_a) : read_mx_file(l, sw.in, sw_a);   // (a copy holds its slot till
                                                                                          // this swap's switch)
        for (int m = 0; m < l.nm && ok; ++m) {
            const Layer& x = member(l, m);
            cudaSetDevice(devs[(size_t) x.dev].id);
            cudaStream_t cs = copy_st[(size_t) x.dev];
            if (l.nm > 1) blob_to_half(l, sw_a, m, sw_h);
            ok = cudaMemcpyAsync(slot_ptr(x, l.spare_vram), l.nm > 1 ? sw_h : sw_a, unit(l),
                                 cudaMemcpyHostToDevice, cs) == cudaSuccess &&
                 cudaStreamSynchronize(cs) == cudaSuccess;
        }
        if (sw.ram >= 0) {   // `out` from its VRAM slot (each member its part) into the spare RAM slot
            bool got = ok;
            for (int m = 0; m < l.nm && got; ++m) {
                const Layer& x = member(l, m);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaStream_t cs = copy_st[(size_t) x.dev];
                got = cudaMemcpyAsync(l.nm > 1 ? sw_h : sw_b, slot_ptr(x, l.slot[(size_t) sw.out]),
                                      unit(l), cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                      cudaStreamSynchronize(cs) == cudaSuccess;
                if (got && l.nm > 1) half_to_blob(l, sw_h, m, sw_b);
            }
            if (got) blob_to_ram(l, sw_b, sw.ram);
            else sw.ram = -1;   // (back to the file's pages then)
        }
        if (check_swaps) {   // DS4_CHECK_SWAPS: the slot (and `out`'s RAM copy) against the model file, byte for byte
            std::vector<uint8_t> dev(l.blob), h(unit(l)), file(l.blob);
            for (int m = 0; m < l.nm && ok; ++m) {
                const Layer& x = member(l, m);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaMemcpy(l.nm > 1 ? h.data() : dev.data(), slot_ptr(x, l.spare_vram), unit(l),
                           cudaMemcpyDeviceToHost);
                if (l.nm > 1) half_to_blob(l, h.data(), m, dev.data());
            }
            ok = ok && read_mx_file(l, sw.in, file.data());
            for (int r = 0; r < 3 && ok; ++r)
                ok = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            bool ok_out = true;
            if (sw.ram >= 0) {
                ok_out = read_mx_file(l, sw.out, file.data());
                ram_to_blob(l, sw.ram, dev.data());
                for (int r = 0; r < 3 && ok_out; ++r)
                    ok_out = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            }
            std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out%s: %s\n", l.il, sw.in, sw.out,
                         sw.ram >= 0 ? " (into RAM)" : "", ok && ok_out ? "matches the model file" : "MISMATCH");
        } else if (!ok) {
            std::fprintf(stderr, "ds4 cache: layer %d expert %d: the move into VRAM failed\n", l.il, sw.in);
        }
    }

    void cache_tick(int T) {
        if (!o.expert_cache || cache_ths.empty()) return;
        // a prompt runs on the arena's bytes (the cache's moves land in the lend slots): nothing to do while its
        // layout is the active one - enter_prefill drains the moves in flight before it switches
        if (phase.load(std::memory_order_acquire) != 1) return;
        std::vector<Swap> ready;
        std::vector<RamSwap> rready;
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            ready.swap(sw_ready);
            rready.swap(ram_ready);
        }
        for (const RamSwap& rs : rready) {   // (nothing reads the tier between windows)
            --ram_inflight;
            Layer& l = L[(size_t) rs.layer];
            if (rs.in < 0 || l.ram[(size_t) rs.in] >= 0 || l.slot[(size_t) rs.in] >= 0) {
                ram_free.push_back(rs.slot);
                continue;
            }
            l.ram[(size_t) rs.in] = rs.slot;
            if (rs.out >= 0 && l.ram[(size_t) rs.out] >= 0) {
                ram_free.push_back(l.ram[(size_t) rs.out]);
                l.ram[(size_t) rs.out] = -1;
            }
            ++st.ram_moves;
        }
        for (const Swap& sw : ready) {
            Layer& l = L[(size_t) sw.layer];
            const int old_slot = l.slot[(size_t) sw.out];
            const std::array<uint8_t*, 4> old_parts = l.hh[(size_t) sw.in];
            if (copy_mode) {   // (`out` takes the RAM place `in` leaves: through the spare, which `in`'s slot becomes)
                const int s_in = l.ram[(size_t) sw.in];
                l.ram[(size_t) sw.in] = -1;
                if (sw.ram >= 0) {
                    l.ram[(size_t) sw.out] = sw.ram;
                    copy_spare[(size_t) l.il] = s_in;
                } else if (s_in >= 0) {   // (the move into RAM failed: `out` back to the file, one copy fewer)
                    --ram_held;
                    if (copy_spare[(size_t) l.il] < 0) copy_spare[(size_t) l.il] = s_in;
                }
            } else if (!l.ram.empty() && l.ram[(size_t) sw.in] >= 0) {   // (in VRAM now: its RAM slot for the next hottest)
                ram_free.push_back(l.ram[(size_t) sw.in]);
                l.ram[(size_t) sw.in] = -1;
            }
            l.slot[(size_t) sw.in] = l.spare_vram;
            if (l.nm == 1) set_vm(l, sw.in, l.spare_vram);
            l.hh[(size_t) sw.in] = {nullptr, nullptr, nullptr, nullptr};
            l.host[(size_t) sw.in] = nullptr;
            l.slot[(size_t) sw.out] = -1;
            if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file has it)
                l.host[(size_t) sw.out] = (uint8_t*) mx_mat(l, sw.out, 0);
            } else if (file_experts) {   // (back to its pages in the file)
                l.hh[(size_t) sw.out] = {file_part(l, sw.out), nullptr, nullptr, nullptr};
                l.host[(size_t) sw.out] = l.hh[(size_t) sw.out][0];
            } else {
                l.hh[(size_t) sw.out] = l.spare_host;
                l.host[(size_t) sw.out] = l.spare_host[0];
                l.spare_host = old_parts;
            }
            l.spare_vram = old_slot;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const Dev& dm = devs[(size_t) x.dev];
                l.tab_pin[2 * m] = (unsigned long long) slot_ptr(x, l.slot[(size_t) sw.in]);
                l.tab_pin[2 * m + 1] = 0ull;
                cudaSetDevice(dm.id);   // (stream order: before the next window)
                cudaMemcpyAsync(x.vram_tab + sw.in, &l.tab_pin[2 * m], 8, cudaMemcpyHostToDevice, dm.st);
                cudaMemcpyAsync(x.vram_tab + sw.out, &l.tab_pin[2 * m + 1], 8, cudaMemcpyHostToDevice, dm.st);
            }
            l.swap_pending = false;
            --sw_inflight;
            ++st.swaps;
        }
        decay_tokens += T;
        if (decay_tokens >= 256) {   // older use counts less: x0.9 every 256 tokens
            decay_tokens = 0;
            for (Layer& l : L)
                for (float& h : l.heat) h *= c_decay;
        }
        if (draining.load(std::memory_order_acquire)) return;   // (a prompt starts: none started here)
        bool queued = false, rqueued = false;
        for (Layer& l : L) {
            if (l.cpu_hits.empty()) continue;
            std::vector<int> hits;
            hits.swap(l.cpu_hits);
            int best = -1, vin = -1;
            for (int e : hits)
                if (l.slot[(size_t) e] < 0 && (best < 0 || l.heat[(size_t) e] > l.heat[(size_t) best])) best = e;
            if (best >= 0 && l.spare_vram >= 0 && !l.swap_pending && sw_inflight < c_inflight) {   // VRAM: the hottest in
                int cold = -1;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0 && (cold < 0 || l.heat[(size_t) e] < l.heat[(size_t) cold])) cold = e;
                if (cold >= 0 && l.heat[(size_t) best] > l.heat[(size_t) cold] * c_margin + c_bonus) {
                    l.swap_pending = true;
                    ++sw_inflight;
                    vin = best;
                    Swap sw{l.il, best, cold};
                    if (copy_mode && l.ram[(size_t) best] >= 0) sw.ram = copy_spare[(size_t) l.il];   // (-1: none)
                    std::lock_guard<std::mutex> lk(sw_mu);
                    sw_queue.push_back(sw);
                    queued = true;
                }
            }
            if (l.ram.empty() || ram_free.empty() || ram_inflight >= kMaxSwaps) continue;
            int rb = -1;   // RAM: the hottest read from the file, in place of the layer's coldest held (or a free slot)
            for (int e : hits)
                if (e != vin && l.slot[(size_t) e] < 0 && l.ram[(size_t) e] < 0 &&
                    (rb < 0 || l.heat[(size_t) e] > l.heat[(size_t) rb]))
                    rb = e;
            if (rb < 0) continue;
            int rc = -1;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.ram[(size_t) e] >= 0 && (rc < 0 || l.heat[(size_t) e] < l.heat[(size_t) rc])) rc = e;
            const bool room = ram_free.size() > 1;   // (slots VRAM freed: filled at once)
            if (!room && !(rc >= 0 && l.heat[(size_t) rb] > l.heat[(size_t) rc] * 1.25f + 1.0f)) continue;
            RamSwap rs;
            rs.layer = l.il;
            rs.in = rb;
            rs.out = room ? -1 : rc;
            rs.slot = ram_free.back();
            ram_free.pop_back();
            ++ram_inflight;
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_queue.push_back(rs);
            rqueued = true;
        }
        if (queued) sw_cv.notify_all();
        if (rqueued) ram_cv.notify_one();
    }

    // every device's work done: the order the layouts change in
    void sync_devs() {
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
            if (di < chunks.size() && chunks[di] != nullptr) cudaStreamSynchronize(chunks[di]->cs);
        }
    }

    // the chunk kernels tile rows (up to a few hundred); a prompt of T rows may touch a little past T, so every
    // row-major buffer's used range is rounded up this many rows - those bytes stay zero, as load left them
    static constexpr int64_t kPrefillRowGuard = 512;

    // ---- the phase switch.  A run of chunks takes the arena for its buffers: the experts lent into it go back to the
    // CPU lane's source (the files - lane_file), the perm bank's spare comes home, the bytes the run uses are zeroed
    // as they were at load.  A bank lying wholly in the buffers' unused tails stays (a prompt of `rows` rows uses only
    // each buffer's first rows), hidden from the prompt's slot map; the refill after the run fills only what went.
    // Both are idempotent; the model's own thread alone calls them, between runs
    bool enter_prefill(int64_t rows, std::string& err) {
        if (phase.load(std::memory_order_acquire) == 0) return true;
        // the cache's moves write into the arena's slots: apply the ones finished, start none, wait for the rest
        draining.store(true, std::memory_order_release);
        bool empty = false;
        {
            const double t0 = now_ms();
            for (;;) {
                cache_tick(0);
                {
                    std::lock_guard<std::mutex> lk(sw_mu);
                    empty = sw_queue.empty() && sw_inflight == 0;
                }
                if (empty || now_ms() - t0 > 30000.0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        draining.store(false, std::memory_order_release);
        if (!empty) {
            std::lock_guard<std::mutex> lk(sw_mu);
            err = "the expert cache did not settle before the prompt (" + std::to_string(sw_inflight) + " swaps in flight)";
            return false;
        }
        phase.store(0, std::memory_order_release);
        st.phase = 0;
        sync_devs();
        // the arena's used ranges for this run: every carved buffer's first fixed + per_row*(rows + guard) bytes.  A
        // chunk touches no byte outside them, so a lend bank wholly inside the rest is kept (and its expert hidden
        // from the prompt's slot map: the prompt must not take it for a resident one)
        std::vector<std::vector<std::array<size_t, 2>>> used(devs.size());
        for (size_t di = 0; di < devs.size(); ++di) {
            auto& u = used[di];
            for (const Dev::ArenaSpan& sp : devs[di].arena_spans) {
                size_t e = sp.fixed;
                if (sp.per_row > 0) e += sp.per_row * (size_t) (std::max<int64_t>(rows, 0) + kPrefillRowGuard);
                e = std::min(e, sp.total);
                if (e > 0) u.push_back({sp.off, sp.off + e});
            }
            std::sort(u.begin(), u.end(),
                      [](const std::array<size_t, 2>& a, const std::array<size_t, 2>& b) { return a[0] < b[0]; });
            std::vector<std::array<size_t, 2>> m;
            for (const std::array<size_t, 2>& r : u) {
                if (!m.empty() && r[0] <= m.back()[1]) m.back()[1] = std::max(m.back()[1], r[1]);
                else m.push_back(r);
            }
            u.swap(m);
        }
        const bool partial = o.expert_lend && o.expert_cache;   // (only then does exit_prefill fill it again)
        int64_t gone = 0, kept = 0;
        double kept_gb = 0;
        for (Layer& l : L) {
            if (!l.lend) continue;
            // the perm bank keeps the spare at its last slot (as place_experts left it): the expert the cache had
            // swapped there goes back to the file with the arena's, leaving that slot free
            const int keep = (o.expert_cache && l.n_perm >= 2 && l.n_perm < l.n_exp) ? l.n_perm - 1 : l.n_perm;
            for (int e = 0; e < l.n_exp; ++e) {
                const int s = l.slot[(size_t) e];
                if (s < keep) continue;
                bool stay = false;
                if (partial && s >= l.n_perm && s < l.n_perm + l.n_lend) {
                    stay = true;
                    for (int m = 0; m < l.nm && stay; ++m) {
                        const Layer& x = member(l, m);
                        const Dev& dm = devs[(size_t) x.dev];
                        const size_t b0 = (size_t) (slot_ptr(x, s) - dm.arena), b1 = b0 + unit(x);
                        for (const std::array<size_t, 2>& r : used[(size_t) x.dev])
                            if (b0 < r[1] && r[0] < b1) {
                                stay = false;
                                break;
                            }
                    }
                }
                l.slot[(size_t) e] = -1;   // (out of the prompt's slot map either way: the lane serves it)
                if (stay) {
                    l.lend_keep.push_back({e, s});
                    ++kept;
                    kept_gb += (double) unit(l) * l.nm / 1e9;
                } else {
                    ++gone;
                }
                if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file's mapping: mx_mat, or the RAM tier's copy)
                    l.host[(size_t) e] = (uint8_t*) mx_mat(l, e, 0);
                } else if (file_experts) {         // (its pages in the file)
                    l.hh[(size_t) e] = {file_part(l, e), nullptr, nullptr, nullptr};
                    l.host[(size_t) e] = l.hh[(size_t) e][0];
                }
            }
            l.n_slots = l.n_perm;
            l.spare_vram = keep < l.n_perm ? keep : -1;
            l.n_vram = l.n_slots - (l.spare_vram >= 0 ? 1 : 0);
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) slot_ptr(x, l.slot[(size_t) e]);
                Dev& dm = devs[(size_t) x.dev];
                cudaSetDevice(dm.id);
                CK(cudaMemcpyAsync(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, dm.st));
            }
        }
        // the bytes this run touches, as load left them: zero (a bank kept in a tail is not touched at all)
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            if (d.arena == nullptr || d.arena_bytes == 0) continue;
            cudaSetDevice(d.id);
            if (used[di].empty()) {
                CK(cudaMemsetAsync(d.arena, 0, d.arena_bytes, d.st));
                continue;
            }
            for (const std::array<size_t, 2>& r : used[di])
                CK(cudaMemsetAsync(d.arena + r[0], 0, r[1] - r[0], d.st));
        }
        sync_devs();
        if (o.verbose)
            std::fprintf(stderr,
                         "ds4 lend: the prompt took the arena back (%lld experts evicted, %lld kept - %.1f GB in the "
                         "buffers' tails)\n",
                         (long long) gone, (long long) kept, kept_gb);
        return true;
    }

    // the decode layout: the arena's lend slots take the hottest experts the CPU lane would otherwise serve, each
    // member its part, a copy stream a device (all of them at once); the cache's spare stays in the perm bank
    bool exit_prefill(std::string& err) {
        if (phase.load(std::memory_order_acquire) == 1) return true;
        // the slots the prompt kept: out of its slot map, never out of the arena - back in the map (the `used` below
        // then leaves them alone, and only the evicted experts are candidates for the refill)
        for (Layer& l : L) {
            for (const std::pair<int, int>& kv : l.lend_keep) {
                if (l.slot[(size_t) kv.first] < 0) l.slot[(size_t) kv.first] = kv.second;
                l.host[(size_t) kv.first] = nullptr;
                l.hh[(size_t) kv.first] = {nullptr, nullptr, nullptr, nullptr};
            }
            l.lend_keep.clear();
        }
        if (!o.expert_lend || !o.expert_cache) {
            phase.store(1, std::memory_order_release);
            return true;
        }
        sync_devs();
        const double t0 = now_ms();
        int64_t filled = 0;
        double bytes = 0;
        std::string cerr;
        for (Layer& l : L) {
            if (!l.lend) continue;
            const int n_tot = std::min(l.n_exp, l.n_perm + l.n_lend);
            const bool perm_spare = o.expert_cache && l.n_perm >= 2 && l.n_perm < l.n_exp;
            const bool keep_spare = o.expert_cache && n_tot >= 2 && n_tot < l.n_exp;
            std::vector<char> used((size_t) std::max(0, n_tot), 0);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0 && l.slot[(size_t) e] < n_tot) used[(size_t) l.slot[(size_t) e]] = 1;
            std::vector<int> frees;
            for (int s = 0; s < n_tot; ++s)
                if (!used[(size_t) s]) frees.push_back(s);
            // the spare is a free slot: the perm bank's last one when the prompt left it (it always does), else the
            // last lend slot the prompt kept out of a bank - it stays empty for the cache's swaps
            int spare = -1;
            if (keep_spare) {
                const int want = perm_spare ? l.n_perm - 1 : n_tot - 1;
                auto it = std::find(frees.begin(), frees.end(), want);
                if (it != frees.end()) {
                    spare = want;
                    frees.erase(it);
                } else if (!frees.empty()) {
                    spare = frees.back();
                    frees.pop_back();
                }
            }
            l.n_slots = n_tot;
            l.spare_vram = spare;
            l.n_vram = n_tot - (spare >= 0 ? 1 : 0);
            // (no `if (frees.empty()) continue`: the tab below must still go out - the slots the prompt kept are back
            // in the map, and the decode kernels read them from it)
            std::vector<int> cands;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) cands.push_back(e);
            std::stable_sort(cands.begin(), cands.end(), [&](int a, int b) { return l.heat[(size_t) a] > l.heat[(size_t) b]; });
            // a GGUF's expert is read into the prompt's pinned staging (idle between runs) a batch at a time by the
            // readers, then DMA'd; else the copy reads the file's mapping (or the RAM tier's copy) as it goes
            const bool mx = l.lay.fmt == gg::kFmtMXFP4;
            const size_t ld = (size_t) member(l, 0).dev;
            Chunk* cst = ld < chunks.size() ? chunks[ld].get() : nullptr;
            uint8_t* stage = cst != nullptr && mx ? cst->stage : nullptr;
            size_t nst = stage != nullptr && cst->readers != nullptr ? (size_t) cst->S : 0;
            // (a blob that outgrows the staging's stride - the drafter's on the card whose trunk experts are smaller:
            // the stride was sized from the trunk's - reads through the mapping or the RAM tier instead, which copies
            // its part straight from a source of its own)
            if (nst > 0 && unit(l) > cst->stage_stride) nst = 0;
            const size_t batch = nst > 0 ? nst : 1;
            const size_t nfill = std::min(frees.size(), cands.size());
            for (size_t i0 = 0; i0 < nfill; i0 += batch) {
                const size_t i1 = std::min(nfill, i0 + batch);
                if (nst > 0) {
                    std::atomic<bool> bad{false};
                    cst->readers->run((int) (i1 - i0), [&](int i) {
                        if (!read_mx(l, cands[i0 + (size_t) i], stage + (size_t) i * cst->stage_stride))
                            bad.store(true);
                    });
                    if (bad.load()) continue;
                }
                for (size_t i = i0; i < i1; ++i) {   // (each member's part of it, on its device's copy stream)
                    const int e = cands[i], s = frees[i];
                    bool ok = true;
                    for (int m = 0; m < l.nm && ok; ++m) {
                        Layer& x = member(l, m);
                        Dev& dm = devs[(size_t) x.dev];
                        cudaSetDevice(dm.id);
                        cerr.clear();
                        ok = stream_expert(l, m, e, slot_ptr(x, s), copy_st[(size_t) x.dev], cerr,
                                           nst > 0 ? stage + (i - i0) * cst->stage_stride : nullptr);
                    }
                    // (before the slots are recorded, and before the staging slots are read into again)
                    for (int m = 0; m < l.nm; ++m) {
                        const size_t xd = (size_t) member(l, m).dev;
                        cudaSetDevice(devs[xd].id);
                        ok = ok && cudaStreamSynchronize(copy_st[xd]) == cudaSuccess;
                    }
                    if (!ok) continue;   // (best effort: the CPU lane serves it)
                    l.slot[(size_t) e] = s;
                    l.host[(size_t) e] = nullptr;
                    l.hh[(size_t) e] = {nullptr, nullptr, nullptr, nullptr};
                    if (l.nm == 1) set_vm(l, e, s);
                    ++filled;
                    bytes += (double) unit(l) * l.nm;
                }
            }
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) slot_ptr(x, l.slot[(size_t) e]);
                Dev& dm = devs[(size_t) x.dev];
                cudaSetDevice(dm.id);
                CK(cudaMemcpyAsync(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, dm.st));
                CK(cudaStreamSynchronize(dm.st));
            }
        }
        sync_devs();
        lend_ms = now_ms() - t0;
        lend_n = filled;
        lend_gb = bytes / 1e9;
        st.lend_n = lend_n;
        st.lend_gb = lend_gb;
        st.lend_ms = lend_ms;
        phase.store(1, std::memory_order_release);
        st.phase = 1;
        if (o.verbose || filled > 0)
            std::fprintf(stderr, "ds4 lend: the decoder took the arena (%lld experts, %.1f GB, filled in %.0f ms)\n",
                         (long long) filled, bytes / 1e9, lend_ms);
        return true;
    }
    // ---- DS4_PASS_LEND (see pl_do / pl_borrow): a layer's map emptied for its pair, its experts back to the CPU lane's
    // source (the files) and its route table zeroed; the caller sets its addressing
    bool pl_clear(Layer& l, std::string& err) {
        for (int e = 0; e < l.n_exp; ++e) {
            l.slot[(size_t) e] = -1;
            l.hh[(size_t) e] = {nullptr, nullptr, nullptr, nullptr};
            if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file's mapping, as enter_prefill hands an evicted one back)
                l.host[(size_t) e] = (uint8_t*) mx_mat(l, e, 0);
            } else if (file_experts) {
                l.hh[(size_t) e] = {file_part(l, e), nullptr, nullptr, nullptr};
                l.host[(size_t) e] = l.hh[(size_t) e][0];
            } else {
                l.host[(size_t) e] = nullptr;   // (no source to hand it back to: never lent)
            }
            if (!l.vm.empty()) l.vm[(size_t) e] = {};
        }
        l.n_slots = 0;
        l.n_vram = 0;
        l.spare_vram = -1;
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
        CK(cudaMemcpyAsync(l.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, d.st));
        CK(cudaStreamSynchronize(d.st));
        return true;
    }
    // the borrower's free slots filled with its hottest non-resident experts - exit_prefill's way: the files' blobs read
    // into the chunk staging by the readers a batch at a time, each DMA'd to its slot and the slot recorded after it
    bool pl_fill(Layer& l, int64_t& filled, double& gb, std::string& err) {
        if (l.n_slots <= 0) return true;
        std::vector<char> used((size_t) l.n_slots, 0);
        for (int e = 0; e < l.n_exp; ++e) {
            const int s = l.slot[(size_t) e];
            if (s >= 0 && s < l.n_slots) used[(size_t) s] = 1;
        }
        std::vector<int> frees;
        for (int s = 0; s < l.n_slots; ++s)
            if (!used[(size_t) s]) frees.push_back(s);
        std::vector<int> cands;
        for (int e = 0; e < l.n_exp; ++e)
            if (l.slot[(size_t) e] < 0) cands.push_back(e);
        std::stable_sort(cands.begin(), cands.end(),
                         [&](int a, int b) { return l.heat[(size_t) a] > l.heat[(size_t) b]; });
        const bool mx = l.lay.fmt == gg::kFmtMXFP4;
        Chunk* cst = (size_t) l.dev < chunks.size() ? chunks[(size_t) l.dev].get() : nullptr;
        uint8_t* stage = cst != nullptr && mx ? cst->stage : nullptr;
        size_t nst = stage != nullptr && cst->readers != nullptr ? (size_t) cst->S : 0;
        if (nst > 0 && unit(l) > cst->stage_stride) nst = 0;
        const size_t batch = nst > 0 ? nst : 1;
        const size_t nfill = std::min(frees.size(), cands.size());
        Dev& d = devs[(size_t) l.dev];
        for (size_t i0 = 0; i0 < nfill; i0 += batch) {
            const size_t i1 = std::min(nfill, i0 + batch);
            if (nst > 0) {   // (the readers: the files' reads, side by side, into the staging)
                std::atomic<bool> bad{false};
                cst->readers->run((int) (i1 - i0), [&](int i) {
                    if (!read_mx(l, cands[i0 + (size_t) i], stage + (size_t) i * cst->stage_stride)) bad.store(true);
                });
                if (bad.load()) {
                    err = "pass-lend: reading layer " + std::to_string(l.il) + "'s experts failed";
                    return false;
                }
            }
            cudaSetDevice(d.id);
            for (size_t i = i0; i < i1; ++i)
                if (!stream_expert(l, 0, cands[i], slot_ptr(l, frees[i]), copy_st[(size_t) l.dev], err,
                                   nst > 0 ? stage + (i - i0) * cst->stage_stride : nullptr))
                    return false;
            CK(cudaStreamSynchronize(copy_st[(size_t) l.dev]));
            for (size_t i = i0; i < i1; ++i) {   // (recorded after the copy: the staging serves the next batch)
                const int e = cands[i], s = frees[i];
                l.slot[(size_t) e] = s;
                l.host[(size_t) e] = nullptr;
                l.hh[(size_t) e] = {nullptr, nullptr, nullptr, nullptr};
                set_vm(l, e, s);
                ++filled;
                gb += (double) unit(l) / 1e9;
            }
        }
        std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
        for (int e = 0; e < l.n_exp; ++e)
            if (l.slot[(size_t) e] >= 0) tab[(size_t) e] = (unsigned long long) slot_ptr(l, l.slot[(size_t) e]);
        cudaSetDevice(d.id);
        CK(cudaMemcpyAsync(l.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, d.st));
        CK(cudaStreamSynchronize(d.st));
        return true;
    }
    // one pass's borrow over the pairs.  Stage 0: each encoder layer takes its tail layer's bank (its own residents stay;
    // the block's other slots are filled); stage 1: each tail layer takes the whole block - its own bank, which pass 1
    // had filled with the encoder layer's experts, and the encoder layer's.  The lender's map is emptied and its slots
    // count 0; the pair's decode addressing is saved the first time and put back by pl_restore
    bool pl_borrow(int stage, std::string& err) {
        if (pl_pairs.empty()) return true;
        const double t0 = now_ms();
        int64_t filled = 0;
        double gb = 0;
        if (pl_saved.empty())   // (their decode addressing, put back by pl_restore)
            for (const PlPair& pr : pl_pairs) {
                pl_saved.push_back({pr.enc, L[(size_t) pr.enc].vram_base, L[(size_t) pr.enc].n_perm});
                pl_saved.push_back({pr.tail, L[(size_t) pr.tail].vram_base, L[(size_t) pr.tail].n_perm});
            }
        for (const PlPair& pr : pl_pairs) {
            Layer& e = L[(size_t) pr.enc];
            Layer& t = L[(size_t) pr.tail];
            Layer& to = stage == 0 ? e : t;
            Layer& from = stage == 0 ? t : e;
            if (!pl_clear(from, err)) return false;   // (its slot count 0: the borrower takes the block)
            from.n_perm = 0;
            to.vram_base = pr.base;   // (the encoder layer's already: the block starts at its bank)
            to.n_perm = stage == 0 ? pr.n_enc : pr.n_tail;
            to.n_slots = to.n_perm;
            to.n_vram = to.n_perm;
            to.spare_vram = -1;
            if (!pl_fill(to, filled, gb, err)) return false;
        }
        if (o.verbose || filled > 0)
            std::fprintf(stderr, "ds4 pass-lend: pass %d, the %s banks lent to the %s: %lld experts, %.1f GB, filled in %.0f ms\n",
                         stage + 1, stage == 0 ? "tail's" : "encoder's", stage == 0 ? "encoder layers" : "tail",
                         (long long) filled, gb, now_ms() - t0);
        return true;
    }
    // the borrows put back: every layer in play cleared (its experts to the files) with its decode addressing; the
    // refill after the prompt (exit_prefill) fills both sides of every pair again from the profile
    bool pl_restore(std::string& err) {
        for (const PlSaved& s : pl_saved) {
            Layer& l = L[(size_t) s.il];
            l.vram_base = s.vram_base;
            l.n_perm = s.n_perm;
            if (!pl_clear(l, err)) return false;
        }
        if (!pl_saved.empty() && o.verbose)
            std::fprintf(stderr, "ds4 pass-lend: the decoder's layout put back\n");
        pl_saved.clear();
        return true;
    }
    bool ensure_decode(std::string& err) {
        if (phase.load(std::memory_order_acquire) == 1) return true;
        return exit_prefill(err);
    }

    // lookahead's thread: the ranges the CPU lane asked for, read in (os::prefetch) off the lane, which computes
    void prefetch_loop() {
        std::vector<std::pair<const void*, size_t>> b;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return quit.load(std::memory_order_relaxed) || !pf_q.empty(); });
                if (quit.load(std::memory_order_relaxed)) return;
                b.swap(pf_q);
            }
            os::prefetch_cold(b);   // (the resident ones would only cost the page walk)
            b.clear();
        }
    }
    // an expert the CPU lane reads from the files (neither in VRAM nor in the RAM tier)
    static bool file_read(const Layer& l, int e) {
        return l.slot[(size_t) e] < 0 && (l.ram.empty() || l.ram[(size_t) e] < 0);
    }
    // the layer whose picks layer l0 guesses a layers ahead (1, 2): the next ones of its stack on its device, MXFP4
    // experts read in place, some of them from the files now (none: nothing to read early - null)
    const Layer* ahead_layer(const Layer& l0, int a) const {
        const int j = l0.il + a, end = l0.mtp ? g.nl + g.n_mtp : g.nl;
        if (j >= end || j >= (int) L.size()) return nullptr;
        const Layer& n = L[(size_t) j];
        if (n.dev != l0.dev || n.mtp != l0.mtp || n.lay.fmt != gg::kFmtMXFP4 || n.gate_w == nullptr || n.n_exp > 1024 ||
            (int) n.slot.size() < n.n_exp)
            return nullptr;
        for (int e = 0; e < n.n_exp; ++e)
            if (file_read(n, e)) return &n;
        return nullptr;
    }
    // lookahead on the CPU lane: request r (layer li, seq) carries the device's guesses for the next layers - the
    // guessed experts read from the files asked for now (each once a window), the guesses kept for the hit counts
    std::vector<std::pair<const void*, size_t>> pf_want;
    void ahead_ask(int li, const Req& r, unsigned seq, int T) {
        pf_want.clear();
        for (int a = 1; a <= 2; ++a) {
            const int m = std::min(r.na[a - 1], a == 1 ? kAhead1 : kPicks), j = li + a;
            if (m <= 0 || j >= (int) L.size()) continue;
            const Layer& n = L[(size_t) j];
            uint8_t* fl = ahead_fl.data() + (size_t) j * 1024;
            uint8_t* rk = ahead_rank.data() + (size_t) j * 1024;
            if (ahead_for[(size_t) j] != seq + (unsigned) a) {
                std::memset(fl, 0, 1024);
                std::memset(rk, 0, 1024);
                ahead_for[(size_t) j] = seq + (unsigned) a;
            }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < m; ++i) {
                    const int e = r.ahead[t][(a - 1) * kAhead1 + i];
                    if (e < 0 || e >= n.n_exp || e >= (int) n.slot.size()) continue;
                    fl[e] |= a == 2 ? 4 : (i < n.k_exp ? 3 : 2);
                    if (a == 1 && (rk[e] == 0 || rk[e] > i + 1)) rk[e] = (uint8_t) (i + 1);
                    if ((fl[e] & 8) || !file_read(n, e)) continue;
                    fl[e] |= 8;
                    for (int q = 0; q < 3; ++q) pf_want.push_back({mx_mat(n, e, q), mx_part_bytes(n, q)});
                    ++st.ahead_pf;
                }
        }
        if (pf_want.empty()) return;
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_q.insert(pf_q.end(), pf_want.begin(), pf_want.end());
        }
        pf_cv.notify_one();
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    void serve_lane() {
        unsigned next = 1;
        auto last = std::chrono::steady_clock::now();
        std::vector<Ent> ents;
        while (!quit.load(std::memory_order_relaxed)) {
            Req& r = reqs[next % kRing];
            if (*(volatile unsigned*) &r.seq != next) {
                cpu_relax();
                if (std::chrono::steady_clock::now() - last > std::chrono::milliseconds(2))
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const int li = layer_of[next % kRing].load(std::memory_order_acquire);
            Resp& rp = resps[next % kRing];
            const int T = std::max(1, std::min(kWin, r.T));
            for (int t = 0; t < T; ++t) rp.cpu[t] = 0;
            if (li >= 0 && li < (int) L.size()) {
                Layer& l = L[(size_t) li];
                uint32_t* pr = prof.data() + (size_t) li * g.n_exp;
                ents.clear();
                // lookahead: the guesses made for this layer (when they are for this request), counted on decode
                const uint8_t* fl = T == 1 && !ahead_for.empty() && ahead_for[(size_t) li] == next
                                        ? ahead_fl.data() + (size_t) li * 1024 : nullptr;
                // the window's CPU-lane picks by expert (up to 8 tokens an entry), their weights by entry row
                float wv[kWin * kPicks];
                int tok_of[kWin * kPicks];
                int nslots = 0;
                std::vector<std::pair<int, int>> picks;   // (expert, t * kPicks + i), CPU-lane only
                const bool dsk = disk_skip_w > 0.0f && !l.mtp && li >= disk_skip_from && li < g.nl;
                for (int t = 0; t < T; ++t) {
                    float wsum = 0.0f;   // (DISK_SKIP_W: the token's route weights)
                    if (dsk)
                        for (int i = 0; i < l.k_exp; ++i)
                            if (r.ids[t][i] >= 0 && r.ids[t][i] < l.n_exp) wsum += r.w[t][i];
                    for (int i = 0; i < l.k_exp; ++i) {
                        const int e = r.ids[t][i];
                        if (e < 0 || e >= l.n_exp) continue;
                        ++pr[e];
                        l.heat[(size_t) e] += 1.0f;
                        if (fl != nullptr) {
                            ++st.ahead_picks;
                            st.ahead_hit_k += fl[e] & 1;
                            st.ahead_hit += (fl[e] & 6) != 0;
                            st.ahead_hit2 += (fl[e] >> 2) & 1;
                        }
                        if (l.slot[(size_t) e] < 0) {
                            l.cpu_hits.push_back(e);
                            if (dsk && r.w[t][i] < disk_skip_w * wsum && !in_ram(l, e)) {
                                ++st.skipped;
                                continue;
                            }
                            if (!l.ram.empty() && l.ram[(size_t) e] >= 0) ++st.ram_experts;
                            else if (fl != nullptr) {
                                ++st.ahead_cold;
                                st.ahead_cold_hit += (fl[e] >> 3) & 1;
                            }
                            picks.push_back({e, t * kPicks + i});
                        }
                    }
                }
                if (fl != nullptr) {   // the guesses by rank: how many were picked
                    const uint8_t* rk = ahead_rank.data() + (size_t) li * 1024;
                    for (int e = 0; e < l.n_exp && e < 1024; ++e) {
                        if (rk[e] == 0 || rk[e] > 16) continue;
                        bool hit = false;
                        for (int i = 0; i < l.k_exp; ++i) hit = hit || r.ids[0][i] == e;
                        ++st.ahead_rank_n[rk[e] - 1];
                        st.ahead_rank_hit[rk[e] - 1] += hit;
                    }
                }
                if (ahead_mode > 0 && !ahead_for.empty()) ahead_ask(li, r, next, T);
                std::sort(picks.begin(), picks.end());
                for (size_t a = 0; a < picks.size();) {
                    Ent en;
                    en.e = picks[a].first;
                    en.slot0 = nslots;
                    en.n = 0;
                    while (a < picks.size() && picks[a].first == en.e && en.n < kc::kExl3CpuMaxN) {
                        const int t = picks[a].second / kPicks, i = picks[a].second % kPicks;
                        en.tok[en.n++] = t;
                        tok_of[nslots] = t;
                        wv[nslots] = r.w[t][i];
                        ++nslots;
                        ++a;
                    }
                    ents.push_back(en);
                }
                // lookahead: the entries asked for early (or in the RAM tier) computed first, so the ones asked for
                // only now get the others' time to come in from the SSD (their slots, so the sums, stay the same)
                int ord[kWin * kPicks];
                const int* po = nullptr;
                if (ahead_order && !ahead_for.empty() && ahead_for[(size_t) li] == next && ents.size() > 1 &&
                    ents.size() <= (size_t) (kWin * kPicks)) {
                    const uint8_t* fa = ahead_fl.data() + (size_t) li * 1024;
                    int n = 0;
                    for (int pass = 0; pass < 2; ++pass)
                        for (size_t q = 0; q < ents.size(); ++q) {
                            const int e = ents[q].e;
                            const bool early = !file_read(l, e) || (fa[e] & 8) != 0;
                            if (early == (pass == 0)) ord[n++] = (int) q;
                        }
                    po = ord;
                }
                if (!ents.empty()) {
                    const double tc = now_ms();
                    cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim, po);
                    const int D = g.dim;
                    for (int t = 0; t < T; ++t) {
                        bool any = false;
                        for (int sl = 0; sl < nslots; ++sl) any = any || tok_of[sl] == t;
                        if (!any) continue;
                        float* o = rp.part[t];
                        for (int e = 0; e < D; ++e) o[e] = 0.0f;
                        for (int sl = 0; sl < nslots; ++sl) {
                            if (tok_of[sl] != t) continue;
                            const float* y = c_yd.data() + (size_t) sl * D;
                            for (int e = 0; e < D; ++e) o[e] += wv[sl] * y[e];
                        }
                        rp.cpu[t] = 1;
                    }
                    st.cpu_ms += now_ms() - tc;
                    st.cpu_experts += nslots;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile unsigned*) &rp.seq = next;
            ++next;
            last = std::chrono::steady_clock::now();
        }
    }

    // a GGUF's matrices (BF16 / MXFP4) to its own kernels, EXL3 ones to x3's
    static bool mv_any(const x3::MvJob* j, int n, cudaStream_t s) {
        bool g = false;
        for (int i = 0; i < n; ++i) g = g || j[i].m->fmt != 0;
        return g ? gg::mv(j, n, s) : x3::mv(j, n, s);
    }
    bool mv(Dev& d, std::initializer_list<x3::MvJob> jobs, std::string& err) {
        x3::MvJob j[x3::kMaxMvJobs];
        int n = 0;
        for (const x3::MvJob& x : jobs) j[n++] = x;
        if (!mv_any(j, n, d.st)) {
            err = "exl3 mv failed";
            return false;
        }
        return true;
    }
    bool mv_n(Dev& d, x3::MvJob* j, int n, std::string& err) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs)
            if (!mv_any(j + a, std::min(x3::kMaxMvJobs, n - a), d.st)) {
                err = "exl3 mv failed";
                return false;
            }
        return true;
    }
    // y[c] = x[c] W for the T activations x + c * x_cs (floats), into y + c * y_cs
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0,
                         float alpha = 1.0f) {
        x3::MvJob j;
        j.m = &m;
        j.xf = x;
        j.y = y;
        j.alpha = alpha;
        j.ncol = T;
        j.xf_cs = x_cs;
        j.y_cs = y_cs;
        return j;
    }

    // ---- the DSpark drafter (DeepSeek's DSparkBlock / forward_spec).  Its blocks' rings hold the trunk's main KV:
    // seed() turns a window's taps (the streams' mean entering layers 37..39) into it, through main_proj / main_norm
    // and each block's wkv / kv_norm / rope / FP8, for the positions the trunk kept.  draft() runs the block [y, noise x
    // (block - 1)] at positions pos .. through the blocks (each query attending to the ring and to the whole block),
    // the trunk's head, the markov bias token by token (greedy) and the confidence head.
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err) {
        if (g.n_mtp == 0 || n <= 0) return true;
        if (n > kWin * kPicks) {
            err = "seeding the drafter with " + std::to_string(n) + " rows";
            return false;
        }
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
        if (dump_d != nullptr) {   // DS4_DSPARK_DUMP: 'S', the first position, the rows, then the taps
            for (Dev& e : devs) {
                cudaSetDevice(e.id);
                CK(cudaStreamSynchronize(e.st));
            }
            cudaSetDevice(d.id);
            const int64_t hdr[4] = {'S', p0, n, (int64_t) nt * D};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(h_taps + (size_t) row0 * nt * D, 4, (size_t) n * nt * D, dump_d);
            std::fflush(dump_d);
        }
        for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
            const int na = std::min(kWin, n - a);
            CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                               cudaMemcpyHostToDevice, d.st));
            if (!mv(d, {job(d_main_proj, d.xe, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
        }
        CK(cudaEventRecord(ev_seed, d.st));
        Finish fm;
        fm.w = d_main_norm;
        fm.eps = g.eps;
        finish(d.ed, n, D, D, fm, d.st);
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (l.dev != d_dev) {
                err = "the DSpark blocks are split across devices";
                return false;
            }
            for (int a = 0; a < n; a += kWin) {
                const int na = std::min(kWin, n - a);
                if (!mv(d, {job(l.wkv, d.ed + (size_t) a * D, d.kvr, na, D, HD)}, err)) return false;
                Finish fk;
                fk.w = l.kv_norm;
                fk.eps = g.eps;
                fk.rd = g.rd;
                fk.pos = (int) (p0 + a);
                fk.freqs = d.freqs_win;
                fk.quant = 1;
                fk.dst = d.ext + (size_t) g.win * HD;
                fk.dst_stride = HD;
                finish(d.kvr, na, HD, HD, fk, d.st);
                ext_to_ring(d.ext, l.win, p0 + a, g.win, na, HD, d.st);
            }
        }
        return true;
    }

    bool draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature, float top_p,
               int top_k, float min_p, uint64_t seed, uint64_t counter) {
        const double t0 = now_ms();
        drafts_q = temperature > 0.0f && sample_drafts;
        dq_temp = temperature;
        dq_top_p = top_p;
        dq_top_k = top_k;
        dq_min_p = min_p;
        dq_seed = seed;
        dq_counter = counter;
        const bool ok = draft_(y, out, conf, err);
        if (!ok) drafts_q = false;
        if (max_drafts > 0 && (int) out.size() > max_drafts) {
            out.resize((size_t) max_drafts);
            conf.resize((size_t) max_drafts);
        }
        st.draft_ms += now_ms() - t0;
        ++st.drafts_rounds;
        return ok;
    }
    bool draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
        out.clear();
        conf.clear();
        const int T = g.d_block, D = g.dim, H = g.hc, V = g.vocab, R = g.d_rank;
        const int64_t p0 = pos;
        if (g.n_mtp == 0 || T < 1 || T > kWin || p0 + T > o.max_context) return true;
        if (!ensure_decode(err)) return false;   // (the drafter's banks are the arena's: a prompt leaves them offline)
        int tk[kWin];
        tk[0] = y;
        for (int t = 1; t < T; ++t) tk[t] = g.d_noise;
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        for (int t = 0; t < T; ++t) std::memcpy(h_emb + (size_t) t * D, embed_pinned + (size_t) tk[t] * D, (size_t) D * 2);
        CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
        embed(d.emb_rows, T, d.h, D, H, d.st);
        init_pre(d.pre, T, d.st);
        WinPlan dp;   // (window attention alone: the mailbox seqs)
        dp.async = !sync_decode && T <= async_max_t;
        for (int i = 0; i < g.n_mtp && dp.async; ++i) {
            const int il = g.nl + i;
            dp.seq[il] = ++seq_issued;
            layer_of[dp.seq[il] % kRing].store(il, std::memory_order_release);
        }
        const size_t hs = (size_t) T * H * D, rs = (size_t) g.win * g.hd;
        if (dump_d != nullptr && h_dump == nullptr &&
            (!halloc(h_dump, (size_t) g.n_mtp * kWin * H * D, err) || !halloc(h_dump_ring, (size_t) g.n_mtp * rs, err)))
            return false;
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (dump_d != nullptr) {   // (stream order: the ring the block reads - as BF16 rows - the stream it leaves)
                ring_to_ext(l.win, d.ext, g.win, g.win, g.hd, d.st);
                CK(cudaMemcpyAsync(h_dump_ring + (size_t) i * rs, d.ext, rs * 2, cudaMemcpyDeviceToHost, d.st));
            }
            if (!layer(l, 0, p0, T, dp, err)) return false;
            if (dump_d != nullptr) {
                cudaSetDevice(d.id);
                CK(cudaMemcpyAsync(h_dump + (size_t) i * hs, d.h, hs * 4, cudaMemcpyDeviceToHost, d.st));
            }
        }
        cudaSetDevice(d.id);
        hc_pre_norm(d.h, d.pre, nullptr, D, H, T, g.eps, d.x, d.st);   // (the confidence head reads it before the norm)
        Dev& hd = devs[(size_t) head_dev];
        // (through the host to the head's device, and to the head group's other members - the drafter's device can be
        // the head group's lead itself, a group of every GPU)
        if (d_dev != head_dev || groups[(size_t) hd.grp].mem.size() > 1) {
            CK(cudaMemcpyAsync(d.bounce, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, d.st));
            CK(cudaEventRecord(d.ev_out, d.st));
        }
        if (d_dev != head_dev) {
            cudaSetDevice(hd.id);
            CK(cudaStreamWaitEvent(hd.st, d.ev_out, 0));
            CK(cudaMemcpyAsync(hd.x, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, hd.st));
        }
        // the drafter's final norm and the trunk's head, on every member of the head group (the others' shares first)
        const std::vector<int>& hg = groups[(size_t) hd.grp].mem;
        const bool mk = mk_split && !drafts_q;   // (sampled drafts: the lead's whole rows, gathered)
        for (int m = (int) hg.size() - 1; m >= 0; --m) {
            Dev& e = devs[(size_t) hg[(size_t) m]];
            cudaSetDevice(e.id);
            if (m > 0) {
                CK(cudaStreamWaitEvent(e.st, d.ev_out, 0));
                CK(cudaMemcpyAsync(e.ao, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, e.st));
            } else
                CK(cudaMemcpyAsync(e.ao, e.x, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, e.st));
            Finish fn;
            fn.w = e.d_norm;
            fn.eps = g.eps;
            finish(e.ao, T, D, D, fn, e.st);
            if (!head_share(e, m, e.ao, T, err, !mk)) return false;
        }
        h_idx[0] = y;
        if (mk) {
            // the markov steps over the group: each member its rows of the bias and its share's argmax, every member
            // then the global one (the sampler's tie rule) and its E row - the same drafts as the lead's whole rows,
            // each step ~1/nm of their 66 MB and no gather of the shares
            const int nm = (int) hg.size();
            const unsigned seq = ++mk_seq;
            for (int m = 0; m < nm; ++m) {
                Dev& e = devs[(size_t) hg[(size_t) m]];
                cudaSetDevice(e.id);
                CK(cudaMemcpyAsync(e.mk_ints, h_idx, 4, cudaMemcpyHostToDevice, e.st));
                CK(cudaMemcpyAsync(e.mk_erow, h_mk_embed + (size_t) y * R, (size_t) R * 2, cudaMemcpyHostToDevice, e.st));
            }
            for (int i = 0; i < T; ++i)
                for (int m = 0; m < nm; ++m) {
                    Dev& e = devs[(size_t) hg[(size_t) m]];
                    cudaSetDevice(e.id);
                    float* row = m == 0 ? hd.logits + (size_t) i * V : e.logits_part + (size_t) i * e.mk_vn;
                    markov_share(row, e.mk_share, e.mk_erow, e.mk_vn, R, e.st);
                    argmax_share(row, e.mk_vn, e.mk_v0, mk_slots + ((size_t) i * nm + m) * 4, seq, e.st);
                    markov_pick(mk_slots + (size_t) i * nm * 4, nm, seq, e.mk_ints + i + 1, h_mk_embed, e.mk_erow, R,
                                e.st);
                }
            cudaSetDevice(hd.id);
        } else {
            CK(cudaMemcpyAsync(d_ints, h_idx, 4, cudaMemcpyHostToDevice, hd.st));
            strata::kernels::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            if (drafts_q) {   // the request's own chain: draft i drawn as the model's token i would be (verify's counter + i)
                sp.greedy = false;
                sp.temperature = dq_temp * (q_temp_scale > 0.0f ? q_temp_scale : 1.0f);
                sp.top_p = dq_top_p;
                sp.top_k = dq_top_k;
                sp.min_p = dq_min_p;
                sp.seed = dq_seed;
            }
            for (int i = 0; i < T; ++i) {   // the markov bias of the token before, then that position's draft
                markov_add(hd.logits + (size_t) i * V, d_mk_embed, d_mk_head, d_ints + i, V, R, hd.st);
                if (drafts_q) {
                    sp.counter = dq_counter + (uint64_t) i;
                    strata::kernels::sample_draft(hd.logits + (size_t) i * V, 1, V, sp, d_specq + i, d_ints + i + 1, hd.st);
                } else
                    strata::kernels::sample_tokens(hd.logits + (size_t) i * V, 1, V, nullptr, 0, sp, d_ints + i + 1, hd.st);
            }
        }
        confidence(hd.x, d_mk_embed, d_ints, d_conf, T, D, R, d_confv, hd.st);
        int tok[kWin + 1];
        float cf[kWin];
        CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        for (int i = 0; i <= T; ++i) tok[i] = h_idx[i];
        for (int i = 0; i < T; ++i) cf[i] = h_w[i];
        if (dump_d != nullptr) {   // 'D', the block's position, y, T, then the drafts, their confidences, the hidden
            std::vector<float> xs((size_t) T * D);   // rows before the final norm; then each block's ring and stream
            CK(cudaMemcpy(xs.data(), hd.x, xs.size() * 4, cudaMemcpyDeviceToHost));
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
            const int64_t hdr[4] = {'D', p0, y, T};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(tok + 1, 4, (size_t) T, dump_d);
            std::fwrite(cf, 4, (size_t) T, dump_d);
            std::fwrite(xs.data(), 4, xs.size(), dump_d);
            for (int i = 0; i < g.n_mtp; ++i) {
                std::fwrite(h_dump_ring + (size_t) i * rs, 2, rs, dump_d);
                std::fwrite(h_dump + (size_t) i * hs, 4, hs, dump_d);
            }
            std::fflush(dump_d);
            cudaSetDevice(hd.id);
        }
        for (int i = 0; i < T; ++i) {
            out.push_back(tok[i + 1]);
            conf.push_back(cf[i]);
        }
        return true;
    }

    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::kernels::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        sp.logit_bias = bias;
        strata::kernels::sample_tokens(d.logits + (size_t) row * g.vocab, 1, g.vocab, nullptr, 0, sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    // [y, drafts] at positions pos ..: the trunk's verdict, its next token, and the state back to what it kept.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: the speculative-sampling rule over the request's own
    // chain (top_k, top_p, min_p, temperature): draft d stands with probability min(1, p(d) / q(d)), q the distribution
    // it was drawn from (the drafter's argmax: a point at d, so p(d)), a rejected one is replaced by a draw from the
    // residual max(0, p - q) - so each kept token is distributed as the trunk's own draw.
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err) {
        const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
        const int T = n + 1, V = g.vocab;
        const int64_t p0 = pos;
        int tk[kWin];
        tk[0] = y;
        for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
        spec_win = true;
        const double tv = now_ms();
        const bool ok = forward(tk, T, T, err, 0);
        st.verify_ms += now_ms() - tv;
        spec_win = false;
        if (!ok) return false;
        Dev& hd = devs[(size_t) head_dev];
        cudaSetDevice(hd.id);
        int m = 0;
        if (temperature <= 0.0f) {
            strata::kernels::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            strata::kernels::sample_tokens(hd.logits, T, V, nullptr, 0, sp, d_ints + 16, hd.st);   // (the T rows at once)
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = h_idx[m];
        } else {
            // row i keeps draft i with probability min(1, p(d) / q(d)), else draws from the residual max(0, p - q)
            // (never d itself): each row a token distributed as the model's own draw at that position, p through the
            // request's chain - the top_k / top_p / min_p cut included (the drafter's argmax can lie outside it: p(d) 0,
            // never kept); past the first rejection the rows do not count
            const bool own = drafts_q && dq_temp == temperature && dq_top_p == top_p && dq_top_k == top_k &&
                             dq_min_p == min_p && dq_seed == seed && dq_counter == counter;
            if (!own) {   // the drafter's argmax: q a point at each draft
                for (int i = 0; i < n; ++i) {
                    h_specq[(size_t) i].n = 1;
                    h_specq[(size_t) i].ids[0] = drafts[(size_t) i];
                    h_specq[(size_t) i].p[0] = 1.0;
                }
                CK(cudaMemcpyAsync(d_specq, h_specq.data(), (size_t) n * sizeof(strata::kernels::SpecList),
                                   cudaMemcpyHostToDevice, hd.st));
            }
            for (int i = 0; i < n; ++i) h_idx[i] = drafts[(size_t) i];
            CK(cudaMemcpyAsync(d_ints + 32, h_idx, (size_t) n * 4, cudaMemcpyHostToDevice, hd.st));
            strata::kernels::SamplerParams sp;
            sp.greedy = false;
            sp.temperature = temperature;
            sp.top_p = top_p;
            sp.top_k = top_k;
            sp.min_p = min_p;
            sp.seed = seed;
            sp.counter = counter;
            strata::kernels::sample_verify(hd.logits, n, V, sp, d_specq, d_ints + 32, d_ints + 16, hd.st);
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) n * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = m < n ? h_idx[m] : sample_row(n, temperature, top_p, top_k, min_p, seed, counter + (uint64_t) n, nullptr);
        }
        drafts_q = false;
        rollback(p0, T, m + 1);
        if (!seed_drafter(p0, m + 1, 0, err)) return false;
        accepted = m;
        st.drafts_offered += n;
        st.drafts_accepted += m;
        for (int i = 0; i < n && i <= m && i < 8; ++i) {
            ++st.draft_reached[i];
            if (i < m) ++st.draft_kept[i];
        }
        return true;
    }

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep) {
        if (keep >= T) return;
        // the ring rows: one launch a device for all its layers (a launch a layer and member was ~300 on 8-way tensor
        // parallel, a millisecond of the host's before the next draft)
        std::vector<RingBatch> rb(devs.size());
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                RingBatch& b = rb[(size_t) l.dev];
                if (b.n == kMaxRingBatch) {   // (more layers on a device than a batch takes: this one alone)
                    cudaSetDevice(devs[(size_t) l.dev].id);
                    ring_restore(l.win, l.ring_bak, p0, g.win, keep, T, g.hd, devs[(size_t) l.dev].st);
                    continue;
                }
                b.ring[b.n] = l.win;
                b.bak[b.n++] = l.ring_bak;
            }
        for (size_t di = 0; di < devs.size(); ++di)
            if (rb[di].n > 0) {
                cudaSetDevice(devs[di].id);
                ring_restore_batch(rb[di], p0, g.win, keep, T, g.hd, devs[di].st);
            }
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                Dev& d = devs[(size_t) l.dev];
                if (l.c_snap != nullptr) {
                    cudaSetDevice(d.id);
                    const size_t ns = (size_t) l.ratio * g.hd * 4;
                    cudaMemcpyAsync(l.c_kv, l.c_snap, ns, cudaMemcpyDeviceToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, l.c_snap + (size_t) l.ratio * g.hd, ns, cudaMemcpyDeviceToDevice, d.st);
                    compress_win(l.c_rows, l.c_rows + (size_t) kWin * g.hd, l.c_kv, l.c_score, l.ratio, (int) p0, keep,
                                 g.hd, l.c_norm, g.eps, d.latent, d.st);
                }
            }
        pos = p0 + keep;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
    }

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) { return forward(&token, 1, want_logits ? 1 : 0, err); }


    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches)
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        Chunk& c = *chunks[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        prof_mark(d, 0);
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ab = groups[(size_t) L0.grp].arb;
        // (temporary: the launch after which a fault first shows, named where it happens)
        auto rp_gle = [&](const char* what) -> bool {
            if (std::getenv("DS4_RP_CHECK") == nullptr) return true;
            const cudaError_t se = cudaDeviceSynchronize();   // (temporary: every stream of the device, so the fault is the launch before)
            const cudaError_t e = se != cudaSuccess ? se : cudaGetLastError();
            if (e != cudaSuccess) {
                std::fprintf(stderr, "ds4 rpbad: %s layer %d dev %d T %d: %s\n", what, l.il, d.id, T,
                             cudaGetErrorString(e));
                err = std::string("rowsplit after ") + what + " (layer " + std::to_string(l.il) + " dev " +
                      std::to_string(l.dev) + " T " + std::to_string(T) + "): " + cudaGetErrorString(e);
                return false;
            }
            std::fprintf(stderr, "ds4 rpok: %s layer %d dev %d\n", what, l.il, d.id);   // (temporary)
            return true;
        };
        if (!rp_gle("layer entry")) return false;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        // Engram, in slabs of 256 tokens
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            const int slot = wp.eg_slot >= 0 ? wp.eg_slot : wp.k % 2;   // (read ahead by the run: two in flight)
            const int efmt = eg_fmt(l.eg);
            uint8_t* host = c.eg_host[slot] + (size_t) l.eg * c.C * per;
            CK(cudaMemcpyAsync(c.eg_rows, host, (size_t) T * per, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(c.eg_ev[slot], s));
            for (int s0 = 0; s0 < T; s0 += 256) {
                const int ns = std::min(256, T - s0);
                engram_dequant_tok(c.eg_rows + (size_t) s0 * per, per, cols, ns, ehd, c.eg_x, s, efmt);
                x3::rows_f16(c.eg_x, cols * ehd, nullptr, ns, cols * ehd, c.eg_x16, s);
                if (!gemm(d, c, l.eg_wkv, c.eg_x16, cols * ehd, ns, c.eg_kv, (H + 1) * D, 0.0f, err)) return false;
                engram_gate(c.h + (size_t) s0 * H * D, c.eg_kv, l.eg_w, H, D, ns, g.eps, s);
            }
        }
        // attention
        if (!hc_mix_chunk(d, c, c.h, l.hca_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, c.a_pre, c.a_post, c.a_comb, s);
        if (!rp_gle("head hc_split")) return false;   // (temporary)
        hc_pre_norm(c.h, c.pre, l.attn_norm, D, H, T, g.eps, c.x, s);
        if (!rp_gle("head hc_pre_norm")) return false;   // (temporary)
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        if (!rp_gle("head rows_f16")) return false;   // (temporary)
        if (!gemm(d, c, l.wq_a, c.x16, D, T, c.qr, g.q_lora, 0.0f, err) ||
            !gemm(d, c, l.wkv, c.x16, D, T, c.kvr, HD, 0.0f, err))
            return false;
        if (!rp_gle("gemm wq_a kv")) return false;   // (temporary)
        if (l.kv_src && !gemm(d, c, l.c_wkv, c.x16, D, T, c.ckr, HD, 0.0f, err)) return false;
        if (!rp_gle("gemm c_wkv")) return false;   // (temporary)
        if (l.kv_src && l.ratio > 1 && !gemm(d, c, l.c_wgate, c.x16, D, T, c.cgr, HD, 0.0f, err)) return false;
        if (!rp_gle("gemm c_wgate")) return false;   // (temporary)
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        finish(c.qr, T, g.q_lora, g.q_lora, fq, s);
        if (!rp_gle("finish qr")) return false;   // (temporary)
        x3::rows_f16(c.qr, g.q_lora, nullptr, T, g.q_lora, c.qr16, s);
        if (!rp_gle("rows_f16 qr")) return false;   // (temporary)
        // the window rows: the ring's (positions p0 - win ..) then the chunk's
        ring_to_ext(l.win, c.ext, p0, g.win, HD, s);
        if (!rp_gle("ring_to_ext")) return false;   // (temporary)
        Finish fk;
        fk.w = l.kv_norm;
        fk.eps = g.eps;
        fk.rd = g.rd;
        fk.pos = (int) p0;
        fk.freqs = freqs;
        fk.quant = 1;
        fk.dst = c.ext + (size_t) g.win * HD;
        fk.dst_stride = HD;
        finish(c.kvr, T, HD, HD, fk, s);
        if (!rp_gle("finish kvr")) return false;   // (temporary)
        // feed_mark's checkpoint: the ring as it was before row snap_t - the old ring, then the chunk's rows before it
        const bool snap = wp.snap_t >= 0 && mi == 0 && L0.snap_win != nullptr;
        if (snap) {
            CK(cudaMemcpyAsync(L0.snap_win, l.win, (size_t) g.win * win_rb(), cudaMemcpyDeviceToDevice, s));
            if (wp.snap_t > 0) ext_to_ring(c.ext, L0.snap_win, p0, g.win, wp.snap_t, HD, s, nullptr);
        }
        ext_to_ring(c.ext, l.win, p0, g.win, T, HD, s, nullptr);
        if (!rp_gle("ext_to_ring")) return false;   // (temporary)
        const uint8_t* ckv = nullptr;
        const long long ss = (long long) o.max_context + 64;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            const float* latent = c.latent;
            if (std::getenv("DS4_RP_CHECK") != nullptr)   // (temporary)
                std::fprintf(stderr,
                             "ds4 rpcomp: layer %d dev %d T %d p0 %lld R %d ng %d kv_src %d ix_src %d owns_k %d "
                             "rp %d latent %p lat16 %p norm %p kvst %p scst %p iwk %p/%p fmt %d kn %dx%d ikr %p "
                             "ix_hd %d\n",
                             l.il, d.id, T, (long long) p0, R, ng, (int) l.kv_src, (int) l.ix_src, (int) l.owns_k,
                             (int) c.rp, (const void*) c.latent, (const void*) c.lat16, (const void*) l.c_norm,
                             (const void*) l.c_kv, (const void*) l.c_score, (const void*) l.i_wk.trellis,
                             (const void*) l.i_wk.raw, l.i_wk.fmt, l.i_wk.k, l.i_wk.n, (const void*) c.ikr,
                             g.ix_hd);
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(c.ckr, T, HD, HD, fc, s);
                    latent = c.ckr;
                } else {
                    if (snap && L0.snap_kv != nullptr) {   // (each slot: the latest row before snap_t that holds it)
                        const int64_t ps = p0 + wp.snap_t;
                        for (int sl = 0; sl < R; ++sl) {
                            const int64_t q = ps - 1 - (((ps - 1 - sl) % R) + R) % R;
                            const float* kv = q >= p0 ? c.ckr + (size_t) (q - p0) * HD : l.c_kv + (size_t) sl * HD;
                            const float* sc = q >= p0 ? c.cgr + (size_t) (q - p0) * HD : l.c_score + (size_t) sl * HD;
                            CK(cudaMemcpyAsync(L0.snap_kv + (size_t) sl * HD, kv, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                            CK(cudaMemcpyAsync(L0.snap_sc + (size_t) sl * HD, sc, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                        }
                    }
                    compress_win(c.ckr, c.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, c.latent, s);
                }
            }
            if (!rp_gle("compress")) return false;   // (temporary)
            const int clen_max = (int) ((p0 + T) / R);
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    x3::rows_f16(latent, HD, nullptr, ng, HD, c.lat16, s);
                    if (!rp_gle("rows_f16 latent")) return false;   // (temporary)
                    if (!gemm(d, c, l.i_wk, c.lat16, HD, ng, c.ikr, g.ix_hd, 0.0f, err)) return false;
                    if (!rp_gle("gemm i_wk")) return false;   // (temporary)
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(c.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                    if (!rp_gle("finish ik")) return false;   // (temporary)
                }
                if (clen_max > 0) {
                    if (!gemm_f16w(c, l.i_wproj, g.ix_heads, D, c.x16, D, T,
                                   1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), c.iw, 64, err))
                        return false;
                    if (!rp_gle("gemm iwproj")) return false;   // (temporary)
                    for (int a0 = 0; a0 < T; a0 += c.AI) {
                        const int na = std::min(c.AI, T - a0);
                        const int64_t pa = p0 + a0;
                        if (!gemm(d, c, l.i_wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.iq,
                                  g.ix_heads * g.ix_hd, 0.0f, err))
                            return false;
                        Finish fi;
                        fi.rd = g.rd;
                        fi.pos = (int) pa;
                        fi.vpp = g.ix_heads;
                        fi.freqs = freqs;
                        fi.quant = 3;
                        finish(c.iq, na * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                        Rows lim;
                        lim.gen = 1;
                        lim.p0 = pa;
                        lim.R = R;
                        lim.kcap = g.ix_topk;
                        lim.ccap = g.cand_topk;
                        lim.blk = std::max(1, g.cand_blk);
                        const int nlast = (int) ((pa + na) / R);   // the slab's widest query
                        int* tk = c.topk + (size_t) a0 * topk_stride();
                        int* cd = c.cand + (size_t) a0 * cand_stride();
                        const float* iw = c.iw + (size_t) a0 * 64;
                        if (nlast <= 0) continue;
                        if (l.uses_cand && (nlast + g.cand_blk - 1) / g.cand_blk > g.cand_topk) {
                            Rows cl = lim;
                            cl.gen = 2;
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, g.cand_topk * g.cand_blk,
                                        cd, (int) cand_stride(), g.cand_blk, cl, c.scores, ss, s);
                            Rows tr = lim;
                            tr.gen = 3;
                            topk(c.scores, ss, na, tr, cd, (int) cand_stride(), g.cand_blk, tk, (int) topk_stride(), s);
                        } else {
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, nlast, nullptr, 0, 1,
                                        lim, c.scores, ss, s);
                            if (l.cand_src) {
                                Rows cr = lim;
                                cr.kcap = g.cand_topk;
                                cand_blocks(c.scores, ss, na, cr, g.cand_blk, c.bm, (long long) bm_stride(), cd,
                                            (int) cand_stride(), s);
                            }
                            topk(c.scores, ss, na, lim, nullptr, 0, 1, tk, (int) topk_stride(), s);
                        }
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(const_cast<float*>(latent), ng, HD, HD, fc, s);
                if (!c.rp && mi == 0 && sr.remote && !(tail_mode && l.il == g.cand_src)) {   // the new rows for the other groups'
                    // readers (slot k % 2, once they took the chunk before last's).  The candidate source's readers run
                    // in the second pass only: their rows are copied to them between the passes, so no put or ack here
                    const int slot = wp.k % 2;
                    if (wp.k >= 2)
                        for (int r : sr.readers)
                            if (!tail_mode || (size_t) r >= pass_devs.size() || pass_devs[(size_t) r])
                                wait_flags(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, 1, 32, wp.win - 2, s);
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.stage(slot);
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.stage(slot) + x_rows(sr) * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.flag(slot), wp.win, d.xcnt, s);
                }
            }
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (!rp_gle("compressor ix")) return false;   // (temporary)
        // (the row split) the chunks behind get this chunk's ring, compressor group and completed rows here: they are
        // final now, and waiting for them instead of for the whole layer leaves only this layer's projections and its
        // indexer between the chunks - not its attention and experts
        if (wp.rp_pub != nullptr) {
            const RpPub& q = *wp.rp_pub;
            xfer_put(q.ring, q.ring_flag, q.ring_seq, d.xcnt, s);
            if (q.comp.n > 0) xfer_put(q.comp, q.comp_flag, q.comp_seq, d.xcnt, s);
            if (q.src.n > 0) xfer_put(q.src, q.src_flag, q.src_seq, d.xcnt, s);
        }
        // the attention in slabs: q, the sparse attention, wo_a, wo_b (this member's share) into acc
        if (NHL == 0) CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * sizeof(*c.acc), s));   // (no heads)
        for (int a0 = 0; a0 < T && NHL > 0; a0 += c.A) {
            const int na = std::min(c.A, T - a0);
            if (!gemm(d, c, l.wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.q, HQ, 0.0f, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) (p0 + a0);
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(c.q, na * NHL, HD, HD, fr, s);
            if (!rp_gle("finish q")) return false;
            AttnRows ar;
            ar.gen = 1;
            ar.p0 = p0;
            ar.t0 = a0;
            ar.win = g.win;
            ar.R = l.ratio;
            ar.kcap = g.ix_topk;
            sparse_attn(c.q, c.ext, ckv, c.topk + (size_t) a0 * topk_stride(), (int) topk_stride(), ar, l.sink, NHL, HD,
                        na, 1.0f / std::sqrt((float) HD), c.o, s, kKvFp4, 0, &d.attn_scr);   // (a short last slab)
            if (!rp_gle("sparse_attn")) return false;
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) (p0 + a0);
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(c.o, na * NHL, HD, HD, fo, s);
            if (!rp_gle("finish o")) return false;
            x3::rows_f16(c.o, HQ, nullptr, na, HQ, c.o16, s);
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                if (!gemm(d, c, l.wo_a[gi], c.o16 + (size_t) gi * per, HQ, na, c.oa + (size_t) gi * g.o_lora,
                          OG * g.o_lora, 0.0f, err))
                    return false;
            x3::rows_f16(c.oa, OG * g.o_lora, nullptr, na, OG * g.o_lora, c.oa16, s);
            if (!gemm(d, c, l.wo_b, c.oa16, OG * g.o_lora, na, c.acc + (size_t) a0 * D, D, 0.0f, err)) return false;
        }
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.a_post, c.a_comb, D, H, T, s);
        if (!rp_gle("attention hc_post")) return false;
        prof_mark(d, 1);
        // the MoE
        if (!hc_mix_chunk(d, c, c.h, l.hcf_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, c.f_pre, c.f_post, c.f_comb, s);
        if (!rp_gle("hc_split")) return false;
        hc_pre_norm(c.h, c.a_pre, l.ffn_norm, D, H, T, g.eps, c.x, s);
        if (!rp_gle("hc_pre_norm")) return false;
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        if (!rp_gle("rows_f16 x")) return false;
        gemv_f32(l.gate_w, c.x, l.n_exp, D, T, c.glog, 1024, s);   // (F32 x: the token path's bits)
        if (!rp_gle("gate gemv")) return false;
        for (int r0 = 0; r0 < T; r0 += kVlRows) {   // (route()'s image bits cover kVlRows rows a call)
            const int nr = std::min(kVlRows, T - r0);
            VlBits vb;
            const bool vl = l.gate_bias_vl != nullptr && vl_bits(p0 + r0, nr, vb);
            route(c.glog + (size_t) r0 * 1024, l.gate_bias, l.n_exp, l.k_exp, nr, g.route_scale,
                  c.ridx + (size_t) r0 * kPicks, c.rw + (size_t) r0 * kPicks, s, l.gate_bias_vl, vl ? &vb : nullptr);
        }
        if (ep_run)   // (the owner's share leaves the other experts' picks out: -1 is in no batch's range)
            CK(cudaMemsetAsync(c.pos, 0xff, (size_t) T * kPicks * 4, s));
        const int n_vs = L0.n_slots + (int) L0.host_list.size();
        CK(cudaMemcpyAsync(l.vs_dev, l.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, s));   // (this member's)
        moe_sort(c.ridx, T, l.k_exp, l.vs_dev, n_vs, c.bounds, c.row_tok, c.pos, s);
        if (!rp_gle("moe_sort")) return false;
        CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (n_vs + 1) * 4, cudaMemcpyDeviceToHost, s));
        if (ep_run) {   // the partners' copy of x | ridx | rw, on its way while this device sorts and computes
            if (!ep_publish(d, c, L0, T, s, err)) return false;
        }
        // the shared expert into acc while the host waits for the bounds
        if (!gemm(d, c, l.s1, c.x16, D, T, c.sg, FF, 0.0f, err) || !gemm(d, c, l.s3, c.x16, D, T, c.su, FF, 0.0f, err))
            return false;
        swiglu(c.sg, c.su, T * FF, g.lim, c.sg, s);
        x3::rows_f16(c.sg, FF, nullptr, T, FF, c.sh16, s);
        if (!gemm(d, c, l.s2, c.sh16, FF, T, c.acc, D, 0.0f, err)) return false;
        prof_mark(d, 2);
        CK(cudaStreamSynchronize(s));
        std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
        if (dump_r != nullptr && mi == 0) {
            const size_t nk = (size_t) T * l.k_exp, np = (size_t) T * kPicks;
            std::vector<int> ri(np), rt(nk);
            std::vector<float> gl((size_t) T * l.n_exp), gb((size_t) l.n_exp);
            CK(cudaMemcpy(ri.data(), c.ridx, np * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rt.data(), c.row_tok, nk * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy2D(gl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                            cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gb.data(), l.gate_bias, gb.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const int64_t hdr[8] = {l.il, p0, T, l.k_exp, l.n_exp, n_vs, L0.n_slots, kPicks};
            std::fwrite(hdr, 8, 8, dump_r);
            std::fwrite(ri.data(), 4, np, dump_r);
            std::fwrite(L0.vs_pin, 4, (size_t) l.n_exp, dump_r);
            std::fwrite(b.data(), 4, b.size(), dump_r);
            std::fwrite(rt.data(), 4, nk, dump_r);
            std::fwrite(gl.data(), 4, gl.size(), dump_r);
            std::fwrite(gb.data(), 4, gb.size(), dump_r);
            std::fflush(dump_r);
        }
        if (mi == 0) {   // the profile (the CPU lane's experts: by the plan's order)
            uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) {
                const int v = L0.vs_pin[e];
                pr[e] += (uint32_t) (b[(size_t) v + 1] - b[(size_t) v]);
            }
            st.experts += (int64_t) T * l.k_exp;
        }
        if (c.flow) {
            if (!flow_layer(d, c, L0, mi, wp.k, T, b, err)) return false;
            if (!rp_gle("flow_layer")) return false;
            if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
            hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
            if (!rp_gle("moe hc_post")) return false;
            CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            if (!rp_gle("layer tail flow")) return false;   // (temporary)
            prof_mark(d, 3);
            return true;
        }
        // the streamed experts as their batches arrive (each batch's slots freed for the copies at once), the VRAM ones
        // in shares between them (the device works while the copies run)
        const int di = l.dev, P = Chunk::kParts;
        int nb = 0;
        for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == wp.k; ++q) ++nb;
        if (ep_run && !copy_ahead(d, c, S, di, err, &L0, b.data(), L0.n_slots)) return false;   // (this layer's picked ranges)
        int v_done = 0;
        for (int bi = 0; bi < nb; ++bi) {
            const Batch& bt = S.b[S.kp];
            const int v0 = L0.n_slots + bt.j0;
            if (ep_run && b[(size_t) v0] == b[(size_t) v0 + bt.n]) {   // (nothing the prompt picks is in this range)
                ++S.kp;
                continue;
            }
            const int half = ep_run ? (int) (S.kc++ % P) : (int) (S.kp % P);
            CK(cudaStreamWaitEvent(s, c.filled[half], 0));
            if (!run_set(d, c, l, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride, c.bounds,
                         b.data(), v0, v0 + bt.n, T, err))
                return false;
            CK(cudaEventRecord(c.freed[half], s));
            ++S.kp;
            if (ep_run && S.out > 0) --S.out;
            if (!copy_ahead(d, c, S, di, err, &L0, b.data(), L0.n_slots)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (bi + 1) / nb);
            if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        if (ep_run && !ep_collect(d, c, L0, T, s, err)) return false;   // (the partners' shares, added in)
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
        CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        if (!rp_gle("layer tail")) return false;   // (temporary)
        prof_mark(d, 3);
        return true;
    }

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams() {
        for (Layer& l : L) {
            if (l.mtp || l.vs_pin == nullptr) continue;
            std::vector<int> all;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) all.push_back(e);
            if (ep_run) {   // expert-parallel: a share a device, in index order (the owner's from its residents on)
                static const int ep_self = std::getenv("DS4_EP_SELF") ? std::atoi(std::getenv("DS4_EP_SELF")) : 0;
                const size_t nd = devs.size();
                for (size_t di = 0; di < nd; ++di) {
                    EpShare& sh = ep_sh[(size_t) l.il][di];
                    sh.list.clear();
                    if (ep_self) {   // (the check's arm: the owner keeps every expert, the partners none)
                        if ((int) di == l.dev) sh.list = all;
                    } else {
                        for (size_t p = di; p < all.size(); p += nd) sh.list.push_back(all[p]);
                    }
                    sh.n = (int) sh.list.size();
                    sh.n_slots = (int) di == l.dev ? l.n_slots : 0;
                    for (int e = 0; e < l.n_exp; ++e) sh.vs_pin[e] = -1;
                    if (sh.n_slots > 0)
                        for (int e = 0; e < l.n_exp; ++e)
                            if (l.slot[(size_t) e] >= 0) sh.vs_pin[e] = l.slot[(size_t) e];
                    for (int j = 0; j < sh.n; ++j) sh.vs_pin[sh.list[(size_t) j]] = sh.n_slots + j;
                }
                const EpShare& own = ep_sh[(size_t) l.il][(size_t) l.dev];   // (the owner's: the layer's plan)
                l.host_list = own.list;
                std::memcpy(l.vs_pin, own.vs_pin, (size_t) l.n_exp * 4);
                continue;
            }
            l.host_list = all;
            for (int m = 0; m < l.nm; ++m) {   // (a group's members: each its own copy)
                Layer& x = member(l, m);
                int j = 0;
                for (int e = 0; e < l.n_exp; ++e) x.vs_pin[e] = l.slot[(size_t) e] >= 0 ? l.slot[(size_t) e] : l.n_slots + j++;
            }
        }
    }

    // ---- expert-parallel prefill (--ep-prefill), the batch path
    // The layer's owner publishes x | ridx | rw to the partners: host-mapped, the copies on its compute stream, the
    // flag set once they are done.  A partner takes them into its chunk, sorts its share of the experts locally and
    // publishes its partial acc the same way; the owner waits on the three flags and adds them (ep_add) before its
    // post.  The partners' expert streams are the same files, one share each, read concurrently.
    bool ep_publish(Dev& d, Chunk& c, const Layer& L0, int T, cudaStream_t s, std::string& err) {
        if (devs.size() <= 1) return true;   // (alone: the share is the whole CPU lane, no partner to tell)
        const size_t o = (size_t) L0.dev;    // (the owner's own buffer: no other device writes it)
        CK(cudaMemcpyAsync(ep_x[o], c.x, (size_t) T * g.dim * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(ep_ridx[o], c.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(ep_rw[o], c.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        set_flag(ep_x_flag[o], ep_base + (unsigned) L0.il + 1u, s);
        c.ep_bytes += (double) T * g.dim * 4 + 2.0 * T * kPicks * 4;
        (void) d;
        return true;
    }
    bool ep_collect(Dev& d, Chunk& c, const Layer& L0, int T, cudaStream_t s, std::string& err) {
        if (devs.size() <= 1) return true;
        const unsigned seq = ep_base + (unsigned) L0.il + 1u;
        for (size_t pj = 0; pj < devs.size(); ++pj) {
            if ((int) pj == L0.dev) continue;
            wait_flags(chunks[pj]->ep_acc_flag, 1, 32, seq, s);
        }
        for (size_t pj = 0; pj < devs.size(); ++pj) {
            if ((int) pj == L0.dev) continue;
            ep_add(c.acc, chunks[pj]->ep_acc, T * g.dim, s);
            c.ep_bytes += (double) T * g.dim * 4;
        }
        (void) d;
        return true;
    }
    // this device's share of layer L0's experts on the owner's behalf: x | ridx | rw in, the picked ranges of the share
    // streamed, the partial acc out.  The share's picked ranges run on the copy stream while this device computes.
    bool ep_partner_turn(int di, Chunk& c, Layer& L0, int k, int T, Streamer& S, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const unsigned seq = ep_base + (unsigned) L0.il + 1u;
        EpShare& sh = ep_sh[(size_t) L0.il][(size_t) di];
        const size_t o = (size_t) L0.dev;   // (the owner's publication)
        {
            const double t0 = now_ms();
            while ((int) (*(volatile unsigned*) ep_x_flag[o] - seq) < 0) {
                if (now_ms() - t0 > 120000.0) {
                    err = "expert-parallel: layer " + std::to_string(L0.il) + "'s input never came";
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        CK(cudaMemcpyAsync(c.x, ep_x[o], (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(c.ridx, ep_ridx[o], (size_t) T * kPicks * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(c.rw, ep_rw[o], (size_t) T * kPicks * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemsetAsync(c.acc, 0, (size_t) T * g.dim * 4, s));   // (the owner's has the shared expert already)
        c.ep_bytes += 2.0 * T * g.dim * 4 + 2.0 * T * kPicks * 4;
        if (sh.n > 0) {
            CK(cudaMemcpyAsync(sh.vs_dev, sh.vs_pin, (size_t) L0.n_exp * 4, cudaMemcpyHostToDevice, s));
            // (a pick of an expert outside the share leaves no row: pos -1 is in no batch's range)
            CK(cudaMemsetAsync(c.pos, 0xff, (size_t) T * kPicks * 4, s));
            moe_sort(c.ridx, T, L0.k_exp, sh.vs_dev, sh.n, c.bounds, c.row_tok, c.pos, s);
            CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (sh.n + 1) * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            const std::vector<int> b(c.bounds_pin, c.bounds_pin + sh.n + 1);
            const int P = Chunk::kParts;
            int nb = 0;
            for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == k; ++q) ++nb;
            if (!copy_ahead(d, c, S, di, err, &L0, b.data(), 0)) return false;   // (this layer's picked ranges)
            for (int bi = 0; bi < nb; ++bi) {
                const Batch& bt = S.b[S.kp];
                const int v0 = bt.j0;   // (a partner's share maps expert list place j to virtual slot j)
                if (b[(size_t) v0] == b[(size_t) v0 + bt.n]) {   // (no row of the prompt picks this range)
                    ++S.kp;
                    continue;
                }
                const int half = (int) (S.kc++ % P);
                CK(cudaStreamWaitEvent(s, c.filled[half], 0));
                if (!run_set(d, c, L0, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride,
                             c.bounds, b.data(), v0, v0 + bt.n, T, err))
                    return false;
                CK(cudaEventRecord(c.freed[half], s));
                ++S.kp;
                if (S.out > 0) --S.out;
                if (!copy_ahead(d, c, S, di, err, &L0, b.data(), 0)) return false;
            }
        }
        CK(cudaMemcpyAsync(c.ep_acc, c.acc, (size_t) T * g.dim * 4, cudaMemcpyDeviceToHost, s));
        set_flag(c.ep_acc_flag, seq, s);
        return true;
    }

    // a chunk's hand-off between groups: h | pre | topk | cand of the producer's chunk of `hand_T` rows (the chunk the
    // host buffer was laid out for), copying its rows [host_row0, host_row0 + T) to/from the destination's rows
    // [dst_row0, dst_row0 + T).  A consumer's chunk spans whole producer chunks and takes each as one piece (the parts
    // nest), so the offsets are always whole rows
    Xfer chunk_xfer_rows(uint8_t* host, int hand_T, Chunk& c, int T, bool put, int host_row0, int dst_row0) const {
        const size_t rb[4] = {(size_t) g.hc * g.dim * 4, 8 * 4, topk_stride() * 4, cand_stride() * 4};
        void* dv[4] = {c.h, c.pre, c.topk, c.cand};
        uint8_t* hv[4] = {host, host + (size_t) hand_T * rb[0], host + (size_t) hand_T * (rb[0] + rb[1]),
                          host + (size_t) hand_T * (rb[0] + rb[1] + rb[2])};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            uint8_t* hp = hv[i] + (size_t) host_row0 * rb[i];
            uint8_t* dp = (uint8_t*) dv[i] + (size_t) dst_row0 * rb[i];
            x.src[i] = put ? (const void*) dp : (const void*) hp;
            x.dst[i] = put ? (void*) hp : (void*) dp;
            x.bytes[i] = (size_t) T * rb[i];
        }
        return x;
    }
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {   // (a producer's: its whole chunk)
        return chunk_xfer_rows(host, T, c, T, put, 0, 0);
    }

    // a run of chunks (start positions p0[k], sizes T[k], seqs seq[k]) on device di: its layers chunk after chunk, the
    // groups pipelined (a group takes chunk k + 1 while the next takes chunk k: the hand-offs and the sources' rows
    // two deep, each slot reused once its readers acknowledged it), the streamed experts' copies running ahead across
    // layers and chunks
    struct Run {
        std::vector<int64_t> p0;
        std::vector<int> T;
        std::vector<unsigned> seq;
        int64_t snap_pos = -1;   // feed_mark: the position whose state before it the layers keep (-1: none)
        int stage = 0;           // 0: the encoder pass (every token); 1: the decoder-tail pass (the last rows only)
    };
    // ---- the decoder tail (DS4_TAIL_PREFILL): layers after the candidate source layer only attend over the last rows
    // they can see - layer L's rows before (nl - 1 - L) * win are not reachable from the last win rows - so a prompt
    // runs them over the last tail_rows() rows only, seeded with the candidate source's h | pre | topk | cand
    bool tail_mode = false;      // (workers: this feed runs the two passes; no source puts to idle readers)
    std::vector<char> pass_devs; // (this pass: the devices with a layer in it - the source puts wait only on them)
    int chunk_prog_dev = -1;     // (the device whose chunks the progress reports follow; -1: none)
    uint8_t* tail_host = nullptr;   // pinned: the tail's h | pre | topk | cand after the candidate source layer
    size_t tail_host_b = 0;
    uint8_t* src_host = nullptr;    // pinned: a source's rows between the passes
    size_t src_host_b = 0;
    int tail_first() const { return g.cand_src >= 0 ? g.cand_src + 1 : -1; }
    int tail_rows() const { return g.cand_src >= 0 ? g.win + (g.nl - 1 - g.cand_src) * g.win : 0; }
    bool tail_feed(const std::vector<Run>& runs, int n, int logits_rows) const {
        static const char* en = std::getenv("DS4_TAIL_PREFILL");
        if (en == nullptr || std::atoi(en) == 0) return false;
        if (g.cand_src < 0 || tail_first() >= g.nl || devs.size() < 2 || logits_rows > kWin) return false;
        if (n <= tail_rows()) return false;
        const int gi = L[(size_t) g.cand_src].grp;
        if (gi >= (int) chunk_steps.size() || chunk_steps[(size_t) gi] < tail_rows()) return false;
        for (size_t g2 = 0; g2 < groups.size(); ++g2) {   // every group running decoder layers needs a chunk that holds the tail
            bool dec = false;
            for (int il = tail_first(); il < g.nl; ++il) dec = dec || L[(size_t) il].grp == (int) g2;
            if (dec && (g2 >= chunk_steps.size() || chunk_steps[g2] < tail_rows())) return false;
        }
        // the head's group must run decoder layers too: pass 1 is what reads the last rows' h and takes their logits
        bool head_dec = false;
        for (int il = tail_first(); il < g.nl; ++il) head_dec = head_dec || L[(size_t) il].grp == devs[(size_t) head_dev].grp;
        if (!head_dec) return false;
        // every decoder layer must read the candidate source's compressed KV (a later KV source would have its own
        // cache, and pass 1 never fills it for the decoder rows copied between the passes)
        for (int il = tail_first(); il < g.nl; ++il)
            if (L[(size_t) il].src >= 0 && srcs[(size_t) L[(size_t) il].src].layer != g.cand_src) return false;
        return runs[(size_t) gi].T.back() >= tail_rows();   // (the source layer's last chunk holds the whole tail)
    }
    bool tail_alloc(std::string& err) {
        if (tail_host == nullptr && !halloc(tail_host, chunk_hand_bytes(tail_rows()), err)) return false;
        tail_host_b = chunk_hand_bytes(tail_rows());
        return true;
    }
    // the candidate source's rows [0, rows) copied to every device that has decoder layers: the decoder pass reads
    // them in place instead of pulling each chunk over the pipelined staging
    bool tail_sync_src(int64_t rows, std::string& err) {
        if (g.cand_src < 0) return true;
        int ci = -1;   // (srcs is the compact list of the KV sources: find the candidate's entry by its layer)
        for (size_t i = 0; i < srcs.size(); ++i)
            if (srcs[i].layer == g.cand_src) { ci = (int) i; break; }
        if (ci < 0) {
            err = "internal: the decoder tail's source layer has no cache";
            return false;
        }
        Src& sr = srcs[(size_t) ci];
        const int from = L[(size_t) g.cand_src].dev;
        rows = std::min<int64_t>(rows / std::max(1, sr.ratio), sr.cap);
        if (rows <= 0) return true;
        const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
        if (src_host == nullptr || src_host_b < cb + ib) {
            if (!halloc(src_host, cb + ib, err)) return false;
            src_host_b = cb + ib;
        }
        Dev& d = devs[(size_t) from];
        cudaSetDevice(d.id);
        CK(cudaStreamSynchronize(d.st));
        if (sr.ckv[(size_t) from] != nullptr)
            CK(cudaMemcpyAsync(src_host, sr.ckv[(size_t) from], cb, cudaMemcpyDeviceToHost, d.st));
        if (sr.ik[(size_t) from] != nullptr)
            CK(cudaMemcpyAsync(src_host + cb, sr.ik[(size_t) from], ib, cudaMemcpyDeviceToHost, d.st));
        CK(cudaStreamSynchronize(d.st));
        for (int di : sr.readers) {   // (the devices outside the source's group: the same-group members compute it themselves)
            bool dec = false;
            for (int il = tail_first(); il < g.nl && !dec; ++il)
                for (int m = 0; m < L[(size_t) il].nm && !dec; ++m)
                    dec = member(L[(size_t) il], m).dev == di;
            if (!dec) continue;   // (no decoder layer here: it will not read the source in the tail pass)
            Dev& o = devs[(size_t) di];
            cudaSetDevice(o.id);
            if (sr.ckv[(size_t) di] != nullptr) CK(cudaMemcpyAsync(sr.ckv[(size_t) di], src_host, cb, cudaMemcpyHostToDevice, o.st));
            if (sr.ik[(size_t) di] != nullptr) CK(cudaMemcpyAsync(sr.ik[(size_t) di], src_host + cb, ib, cudaMemcpyHostToDevice, o.st));
            CK(cudaStreamSynchronize(o.st));
        }
        cudaSetDevice(d.id);
        return true;
    }

    // ---- live telemetry: what each device is doing (the Monitor's Prefill and Loan windows), at most four lines a
    // second a device.  The server takes LIVE lines in its reader thread; a request's stream never sees them.
    struct LiveDev {
        std::atomic<int64_t> stamp{0};
        std::atomic<int> stage{0}, il{-1}, k{0}, nk{0};
        std::atomic<int64_t> p0{0}, done{0}, total{0};
        std::atomic<int> T{0};
    };
    std::unique_ptr<LiveDev[]> live_dev;
    std::atomic<bool> live_on{false};
    std::mutex live_mu;
    int64_t live_base = 0, live_ntot = 0;   // (the feed's first token and length; written before the workers start)
    void live_line(int di, int stage, int il, int k, int nk, int64_t p0, int T, const char* path = "flow",
                   int grp = -1, int ngrp = 0, int wait = -1, int chunk_rows = -1, int round = 0, int nround = 1) {
        if (!live_on.load(std::memory_order_relaxed) || !live_dev) return;
        if (di < 0 || (size_t) di >= devs.size()) return;
        LiveDev& v = live_dev[(size_t) di];
        const int64_t t = (int64_t) now_ms();
        int64_t last = v.stamp.load(std::memory_order_relaxed);
        if (t - last < 250) return;
        if (!v.stamp.compare_exchange_strong(last, t)) return;   // (another layer of the same device is printing)
        v.stage.store(stage, std::memory_order_relaxed);
        v.il.store(il, std::memory_order_relaxed);
        v.k.store(k, std::memory_order_relaxed);
        v.nk.store(nk, std::memory_order_relaxed);
        v.p0.store(p0, std::memory_order_relaxed);
        v.T.store(T, std::memory_order_relaxed);
        const int64_t done = stage == 2 ? 0 : std::max<int64_t>(0, std::min<int64_t>(live_ntot, p0 + T - live_base));
        v.done.store(done, std::memory_order_relaxed);
        v.total.store(stage == 2 ? 0 : live_ntot, std::memory_order_relaxed);
        const Dev& d = devs[(size_t) di];
        int perm = 0, lend = 0, mtp = 0;
        for (const Layer& l : L)
            for (int m = 0; m < l.nm; ++m) {
                const Layer& x = member(l, m);
                if (x.dev != di) continue;
                if (l.mtp) mtp += l.n_lend;
                else {
                    perm += x.n_perm;
                    lend += x.n_lend;
                }
            }
        size_t fr = 0, tot = 0;
        cudaSetDevice(d.id);
        cudaMemGetInfo(&fr, &tot);
        const char* sn = stage == 1 ? "tail" : stage == 2 ? "decode" : "prompt";
        const int ch = d.grp < (int) chunk_steps.size() ? chunk_steps[(size_t) d.grp] : 0;
        char ext[128] = "";
        if (path != nullptr && std::strcmp(path, "rowsplit") == 0) {   // (the mode's own keys, nothing renamed)
            int eo = std::snprintf(ext, sizeof ext, " grp=%d ngrp=%d", grp, ngrp);
            if (wait >= 0) eo += std::snprintf(ext + eo, sizeof ext - (size_t) eo, " wait=%d", wait);
            if (chunk_rows >= 0)
                eo += std::snprintf(ext + eo, sizeof ext - (size_t) eo, " chunk_rows=%d", chunk_rows);
            if (nround > 1)
                std::snprintf(ext + eo, sizeof ext - (size_t) eo, " round=%d nround=%d", round, nround);
        }
        {
            std::lock_guard<std::mutex> lk(live_mu);
            std::fprintf(stdout,
                         "LIVE dev=%d stage=%s path=%s layer=%d k=%d nk=%d p0=%lld T=%d done=%lld total=%lld "
                         "arena=%zu perm=%d lend=%d mtp=%d free=%zu chunk=%d%s\n",
                         d.id, sn, path, il, k, nk, (long long) p0, T, (long long) done,
                         (long long) (stage == 2 ? 0 : live_ntot),
                         d.arena_bytes >> 20, perm, lend, mtp, fr >> 20, ch, ext);
            std::fflush(stdout);
        }
    }

    bool run_dev_layers(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;   // (none: the flow streams)
        const size_t hb = chunk_hand_bytes(c.C);
        // the Engram rows: (layer, chunk) after (layer, chunk) in the run's order, each read on a helper thread while
        // the one before runs (slot u % 2, once unit u - 2's copy of it is done)
        std::vector<std::pair<int, int>> egu;   // (Engram layer, chunk)
        for (int il : c.fl_layers)
            if (L[(size_t) il].eg >= 0)
                for (int k = 0; k < nk; ++k) egu.push_back({L[(size_t) il].eg, k});
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egu, eg_per](size_t u) {
            if (u >= 2) cudaEventSynchronize(c.eg_ev[u % 2]);
            const int li = egu[u].first, k = egu[u].second;
            engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[u % 2] + (size_t) li * c.C * eg_per);
        };
        size_t eg_u = 0;
        std::future<void> eg_next;
        if (!egu.empty()) eg_next = std::async(std::launch::async, read_eg, (size_t) 0);
        for (size_t li = 0; li < c.fl_layers.size(); ++li) {
            const int il = c.fl_layers[li];
            Layer& L0 = L[(size_t) il];
            const bool last_l = li + 1 == c.fl_layers.size();
            for (int k = 0; k < nk; ++k) {
                const int64_t p0 = run.p0[(size_t) k];
                const int T = run.T[(size_t) k], slot = k % 2;
                const bool last_k = k + 1 == nk;
                WinPlan wp;
                wp.par = true;
                wp.async = true;
                wp.win = run.seq[(size_t) k];
                wp.k = k;
                wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
                if (L0.eg >= 0) {   // this unit's rows ready, the next one's on their way
                    const double te = now_ms();
                    eg_next.get();
                    st.engram_ms += now_ms() - te;
                    wp.eg_slot = (int) (eg_u % 2);
                    if (++eg_u < egu.size()) eg_next = std::async(std::launch::async, read_eg, eg_u);
                }
                if (li == 0) {   // the chunk's input: its embedding rows
                    if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                    for (int t = 0; t < T; ++t)
                        std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                    (size_t) D * 2);
                    CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    CK(cudaEventRecord(c.emb_ev[slot], s));
                    embed(c.emb, T, c.h, D, H, s);
                    init_pre(c.pre, T, s);
                } else {   // or its state after the layer before
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, false);
                    for (int i = 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
                }
                if (last_k && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                if (!layer_chunk(L0, 0, p0, T, wp, S, err)) return false;
                live_line(di, 0, il, k, nk, p0, T);
                if (dump_l != nullptr) dump_layer(d, c.h, il, p0, T);
                if (!last_l) {   // the chunk's state kept for the next layer
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, true);
                    for (int i = 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, s));
                    continue;
                }
                // through every layer: the chunk's end, and the logits
                if ((size_t) k < chunk_done.size()) {
                    CK(cudaEventRecord(chunk_done[(size_t) k], s));
                    chunk_rec[(size_t) k].store(1, std::memory_order_release);
                }
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
            lm_layers.store((int) li + 1, std::memory_order_release);
        }
        return true;
    }

    bool run_dev_chunks(int di, const std::vector<Run>& runs, std::string& err) {
        const Run& run = runs[(size_t) devs[(size_t) di].grp];   // (this device's group's run of chunks)
        if (run.T.empty()) return true;   // (a pass with no chunk on this device: run_pass submits to every worker)
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        const bool stage1 = run.stage == 1;   // (the decoder tail: layers after the candidate source layer only)
        const bool ep = ep_run && !tail_mode;   // (expert-parallel: every layer, a share a device, the batch path)
        struct FlowBack {   // (EP runs the batch path: the flow's machinery comes back for whatever runs next)
            Chunk& c;
            bool was;
            ~FlowBack() { c.flow = was; }
        } flow_back{c, c.flow};
        if (ep) {
            c.flow = false;
            c.ep_bytes = 0;
        }
        Streamer S;
        const int B = std::max(1, c.ring_n / Chunk::kParts);
        if (c.flow || ep) {   // (the flow decides its streams as the layers route; nothing planned here - EP resets too)
            CK(cudaStreamSynchronize(s));
            CK(cudaStreamSynchronize(c.cs));
            c.items.clear();
            c.base = c.n_read = c.n_dma = c.n_comp = 0;
            c.n_file = 0;
            std::fill(c.stage_item.begin(), c.stage_item.end(), -1);
            c.fl_layers.clear();
            for (int il = 0; il < g.nl; ++il)   // (a group's layers: on each member)
                for (int m = 0; m < L[(size_t) il].nm; ++m)
                    if (member(L[(size_t) il], m).dev == di &&
                        (!tail_mode || (stage1 ? il >= tail_first() : il <= g.cand_src)))
                        c.fl_layers.push_back(il);
            c.fl_T = run.T;
            c.spec_k = c.spec_il = -1;
            c.wait_ms = 0;
            c.it_file = c.it_ram = c.it_spec = c.it_spec_unused = 0;
            if (c.io != nullptr) {
                c.io_d0 = c.io->direct_bytes();
                c.io_b0 = c.io->buffered_bytes();
            }
            if (!ep && !c.fl_layers.empty()) {
                flow_spec(c, 0, c.fl_layers[0], run.T[0]);
                if (!flow_pump(c, err)) return false;
            }
            if (!ep && c.lm) return run_dev_layers(di, run, err);   // (one device: layer-major, the run's chunks in order)
        }
        for (int k = 0; k < nk && !(c.flow && !ep); ++k)
            for (int il = 0; il < g.nl; ++il) {
                if (tail_mode && (stage1 ? il < tail_first() : il > g.cand_src)) continue;
                const Layer& L0 = L[(size_t) il];
                const std::vector<int>* list = nullptr;
                if (ep) {   // (this device's share of every layer: its own, or a partner's)
                    list = &ep_sh[(size_t) il][(size_t) di].list;
                } else {
                    bool here = false;
                    for (int m = 0; m < L0.nm; ++m) here = here || member(L0, m).dev == di;
                    if (!here) continue;
                    list = &L0.host_list;
                }
                for (int j0 = 0; j0 < (int) list->size(); j0 += B)
                    S.b.push_back({k, il, j0, std::min(B, (int) list->size() - j0), list});
            }
        if (!S.b.empty() && c.ring == nullptr) {
            err = "no slots to stream experts into on CUDA" + std::to_string(d.id);
            return false;
        }
        std::atomic<bool> ra_stop{false};
        std::thread ra;
        // (expert-parallel: its shares are read as their routing comes, range by range - no whole-share read-ahead)
        if (file_experts && !S.b.empty() && !ep) ra = std::thread([this, &S, di, &ra_stop] { read_ahead(S, di, ra_stop); });
        struct Join {   // (every way out of the run: the read-ahead ends before S does)
            std::atomic<bool>& stop;
            std::thread& th;
            ~Join() {
                stop.store(true);
                if (th.joinable()) th.join();
            }
        } ra_join{ra_stop, ra};
        if (!copy_ahead(d, c, S, di, err)) return false;
        // this device's Engram layers: chunk k's rows read on a helper thread while chunk k - 1 runs (slot k % 2, once
        // chunk k - 2's copy of it is done)
        std::vector<int> egs;
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di && !l0.mtp && l0.eg >= 0 &&
                    (!tail_mode || (stage1 ? l0.il >= tail_first() : l0.il <= g.cand_src)))
                    egs.push_back(l0.eg);
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egs, eg_per](int k) {
            if (k >= 2) cudaEventSynchronize(c.eg_ev[k % 2]);
            for (int li : egs)
                engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[k % 2] + (size_t) li * c.C * eg_per);
        };
        std::future<void> eg_next;
        if (!egs.empty()) eg_next = std::async(std::launch::async, read_eg, 0);
        for (int k = 0; k < nk; ++k) {
            const int64_t p0 = run.p0[(size_t) k];
            const int T = run.T[(size_t) k], slot = k % 2;
            const bool last_k = k + 1 == nk;
            if (!egs.empty()) {   // chunk k's rows ready, chunk k + 1's on their way
                const double te = now_ms();
                eg_next.get();
                if (d.mi == 0) st.engram_ms += now_ms() - te;
                if (k + 1 < nk) eg_next = std::async(std::launch::async, read_eg, k + 1);
            }
            WinPlan wp;
            wp.par = true;
            wp.async = true;
            wp.win = run.seq[(size_t) k];
            wp.k = k;
            wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
            bool started = false, ran = false;
            std::vector<char> got(srcs.size(), 0);
            for (int il = 0; il < g.nl; ++il) {
                Layer& L0 = L[(size_t) il];
                int mi = -1;
                for (int m = 0; m < L0.nm; ++m)
                    if (member(L0, m).dev == di) mi = m;
                if (mi < 0) {
                    if (ep && !ep_partner_turn(di, c, L0, k, T, S, err)) return false;   // (this device's share)
                    continue;
                }
                if (tail_mode && (stage1 ? il < tail_first() : il > g.cand_src)) continue;
                ran = true;
                if (!started) {   // the chunk's input: the tail's saved rows, its embedding rows, or the hand-off
                    started = true;
                    if (stage1 && il == tail_first()) {
                        if (tail_host == nullptr) {
                            err = "internal: the decoder tail's rows are missing";
                            return false;
                        }
                        const Xfer x = chunk_xfer_rows(tail_host, T, c, T, false, 0, 0);
                        for (int i = 0; i < x.n; ++i)
                            CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
                    } else if (il == 0) {
                        if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                        for (int t = 0; t < T; ++t)
                            std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                        (size_t) D * 2);
                        CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                        CK(cudaEventRecord(c.emb_ev[slot], s));
                        embed(c.emb, T, c.h, D, H, s);
                        init_pre(c.pre, T, s);
                    } else {   // the previous group's hand-off: its chunk(s) covering this one, a piece at a time
                        Chunk& a = *chunks[(size_t) L[(size_t) il - 1].dev];
                        const Run& fr = runs[(size_t) L[(size_t) il - 1].grp];
                        for (size_t fk = 0; fk < fr.T.size(); ++fk) {
                            const int64_t fp = fr.p0[fk];
                            if (fp + fr.T[fk] <= p0 || fp >= p0 + T) continue;
                            const int64_t lo = std::max(fp, p0), hi = std::min<int64_t>(fp + fr.T[fk], p0 + T);
                            const int fslot = (int) (fk % 2);
                            xfer_get(chunk_xfer_rows(a.hand[fslot], fr.T[fk], c, (int) (hi - lo), false,
                                                     (int) (lo - fp), (int) (lo - p0)),
                                     a.hand_flag[fslot], fr.seq[fk], s);
                            set_flag(a.hand_ack + ((size_t) fslot * 8 + mi) * 32, fr.seq[fk], s);
                        }
                    }
                }
                for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                    if (sidx < 0 || got[(size_t) sidx]) continue;
                    got[(size_t) sidx] = 1;
                    if (tail_mode && stage1) continue;   // (the source's rows are in place: copied between the passes)
                    Src& sr = srcs[(size_t) sidx];
                    if (L[(size_t) sr.layer].grp == L0.grp) continue;
                    const Run& ur = runs[(size_t) L[(size_t) sr.layer].grp];
                    const size_t srow = x_rows(sr);
                    for (size_t ks = 0; ks < ur.T.size(); ++ks) {   // the source's chunks this one covers, a piece each
                        const int64_t q = ur.p0[ks];
                        if (q + ur.T[ks] <= p0 || q >= p0 + T) continue;
                        const int64_t lo = std::max(q, p0), hi = std::min<int64_t>(q + ur.T[ks], p0 + T);
                        const int64_t rs = q / sr.ratio, r0 = lo / sr.ratio, r1 = hi / sr.ratio;
                        const int sslot = (int) (ks % 2);
                        Xfer x;
                        if (r1 > r0 && sr.ckv[(size_t) di] != nullptr) {
                            x.src[x.n] = sr.stage(sslot) + (size_t) (r0 - rs) * ckv_rb();
                            x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) r0 * ckv_rb();
                            x.bytes[x.n++] = (size_t) (r1 - r0) * ckv_rb();
                        }
                        if (r1 > r0 && sr.ik[(size_t) di] != nullptr) {
                            x.src[x.n] = sr.stage(sslot) + srow * ckv_rb() + (size_t) (r0 - rs) * ik_rb();
                            x.dst[x.n] = sr.ik[(size_t) di] + (size_t) r0 * ik_rb();
                            x.bytes[x.n++] = (size_t) (r1 - r0) * ik_rb();
                        }
                        if (x.n > 0) xfer_get(x, sr.flag(sslot), ur.seq[ks], s);
                        set_flag(sr.x_ack + ((size_t) sslot * devs.size() + di) * 32, ur.seq[ks], s);   // (taken, or none)
                    }
                }
                if (last_k && mi == 0 && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                if (!layer_chunk(L0, mi, p0, T, wp, S, err)) return false;
                if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, il, p0, T);
                live_line(di, stage1 ? 1 : 0, il, k, nk, p0, T);
                if (!stage1 && tail_mode && L0.il == g.cand_src && mi == 0 && last_k) {   // the tail's input rows
                    if (!tail_alloc(err)) return false;
                    if (T >= tail_rows()) {
                        const Xfer x = chunk_xfer_rows(tail_host, tail_rows(), c, tail_rows(), true, 0, T - tail_rows());
                        for (int i = 0; i < x.n; ++i)
                            CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, s));
                    }
                }
                const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
                const bool next_runs = !tail_mode || (stage1 ? il + 1 >= tail_first() : il + 1 <= g.cand_src);
                if (last && next_runs && mi == 0 && il + 1 < g.nl) {   // the hand-off, once the next group took the chunk before last
                    const int nnext = L[(size_t) il + 1].nm;
                    if (k >= 2) wait_flags(c.hand_ack + (size_t) slot * 8 * 32, nnext, 32, wp.win - 2, s);
                    xfer_put(chunk_xfer(c.hand[slot], c, T, true), c.hand_flag[slot], wp.win, d.xcnt, s);
                }
            }
            if (ran && di == chunk_prog_dev && (size_t) k < chunk_done.size()) {
                CK(cudaEventRecord(chunk_done[(size_t) k], s));
                chunk_rec[(size_t) k].store(1, std::memory_order_release);
            }
            if (ran && d.grp == devs[(size_t) head_dev].grp) {   // the head group: each member its share of the logits
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        if (di != head_dev) continue;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
        }
        return true;
    }

    // ---- the row-split prefill (--row-split-prefill / DS4_ROW_SPLIT): the plan, the stage swaps, the wavefront
    //
    // Every device holds the same K-layer group of replicas and runs its own chunk of the prompt's rows through it,
    // group by group.  A chunk's layer needs what the chunks before it left at that layer - the 128-row window ring,
    // the compressor's partial group, and (at the one layer that writes it) the compressed-KV and index-key rows it
    // completed - so the stages run as a wavefront: a chunk starts layer L once every chunk before it has published
    // L.  Every wait is a device-side spin on a host-mapped flag a chunk strictly before it sets, so the waits go one
    // way only and cannot deadlock.  Nothing of the chunked pipeline changes: this path runs beside it.
    //
    // The split can run the mode at all (load): plain layers of a GGUF pack, one member a group, the head whole, the
    // drafter on the head's device, the flow's in-place experts on every card
    bool rp_static(std::string& why) const {
        if (chunk <= 0) { why = "no chunked prompt path (--prefill-chunk 0)"; return false; }
        if (ep_do()) { why = "expert-parallel prefill is on"; return false; }
        if (!file_experts) { why = "the experts are not read in place"; return false; }
        if (groups.size() != devs.size()) { why = "a tensor-parallel group"; return false; }
        if (head_nm() != 1) { why = "the head is split over a group"; return false; }
        if (d_dev != head_dev) { why = "the drafter's blocks are not on the head's device"; return false; }
        if (std::getenv("DS4_FLOW") != nullptr && std::atoi(std::getenv("DS4_FLOW")) == 0) {
            why = "DS4_FLOW=0 (the batch path)";
            return false;
        }
        for (const Layer& l : L) {
            if (l.mtp) continue;
            if (l.nm != 1 || !l.mates.empty()) {
                why = "layer " + std::to_string(l.il) + " is in a tensor-parallel group";
                return false;
            }
            if (l.lay.fmt != gg::kFmtMXFP4) {
                why = "layer " + std::to_string(l.il) + " is not a GGUF layer";
                return false;
            }
            if (l.kv_src && l.ik_src != l.src) {
                why = "layer " + std::to_string(l.il) + " has a separate index source";
                return false;
            }
            if (l.kv_src && !l.ix_src) {   // (its index keys are written by the layer that writes its compressed KV)
                why = "layer " + std::to_string(l.il) + " writes no index keys";
                return false;
            }
            if (l.dense_bytes == 0) {
                why = "layer " + std::to_string(l.il) + " has no dense bytes";
                return false;
            }
        }
        if (chunks.size() != devs.size()) { why = "no chunk buffers"; return false; }
        return true;
    }
    // the share of what the split's reserve leaves that the replicas may take (DS4_ROW_SPLIT_FRAC, 0.5)
    double rp_frac() const { return std::min(1.0, std::max(0.05, rp_env_f("DS4_ROW_SPLIT_FRAC", 0.5))); }

    // a slot's arena: one cudaMalloc the swaps carve the stage's dense weights from (zeroed once - dalloc's bump
    // branch never zeroes, every carve is fully written by its upload)
    bool rp_arena(Dev& d, RpRep& R, size_t bytes, std::string& err) {
        cudaSetDevice(d.id);
        void* v = nullptr;
        const cudaError_t e = cudaMalloc(&v, bytes);
        if (e != cudaSuccess) {
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            err = "the row-split replica arena (" + std::to_string(bytes >> 20) + " MiB) on CUDA" +
                  std::to_string(d.id) + " failed: " + cudaGetErrorString(e) + " (free " + std::to_string(fr >> 20) +
                  " MiB)";
            return false;
        }
        if (cudaMemsetAsync(v, 0, bytes, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            cudaFree(v);
            err = "zeroing the row-split replica arena on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        d.allocs.push_back(v);
        d.caches += bytes;
        R.bump.base = (uint8_t*) v;
        R.bump.bytes = bytes;
        R.bump.off = 0;
        R.bump.owner = d.id;
        return true;
    }

    // the plan, once at load: K layers a group (the same on every device - the tightest card decides), the groups,
    // every stage swap's arena and the wavefront's buffers.  With the mode off, or the room missing, rp.on stays
    // false and every prompt takes today's path
    bool rp_plan(std::string& err) {
        if (!rp_do()) return true;
        std::string why;
        if (!rp_static(why)) {
            std::fprintf(stderr, "ds4 rowsplit: off (%s)\n", why.c_str());
            return true;
        }
        const int nl = g.nl;
        // the room a device has for replicas: a share of what the split's reserve leaves now that its dense weights and
        // the chunk buffers are down (the experts it does not get are the price of the copies; the rest stays decode's)
        const double frac = rp_frac();
        const int budget_mb = rp_env_int("DS4_ROW_SPLIT_BUDGET_MB", 0);
        std::vector<size_t> room(devs.size(), 0);
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const int64_t after = std::max<int64_t>(0, (int64_t) fr - (int64_t) resv[di]);
            size_t r = (size_t) (frac * (double) after);
            if (budget_mb > 0) r = std::min(r, (size_t) budget_mb << 20);
            room[di] = r;
        }
        // a slot: the arena of the largest layer it holds (dense_bytes plus a margin for the tensors it does not
        // count) and its state - the ring, and the compressor's partial group where the slot's layers have one
        auto slot_of = [&](int j, int k) {
            size_t m = 0;
            for (int il = j; il < nl; il += k) m = std::max(m, L[(size_t) il].dense_bytes);
            return align256(m) + (8u << 20);
        };
        auto state_of = [&](int j, int k) {
            size_t b = (size_t) g.win * win_rb();
            for (int il = j; il < nl; il += k) {
                const Layer& l = L[(size_t) il];
                if (l.kv_src && l.ratio > 1) b += 2 * (size_t) l.ratio * g.hd * 4;
            }
            return b;
        };
        const int min_k = std::max(1, rp_env_int("DS4_ROW_SPLIT_MIN_K", 4));
        int k = 0;
        for (int c = std::min(nl, std::max(1, rp_env_int("DS4_ROW_SPLIT_K", nl))); c >= min_k; --c) {
            size_t need = 2 * (size_t) g.n_exp * 4;   // (the virtual-slot maps: pinned, and the device's copy)
            for (int j = 0; j < c; ++j) need += slot_of(j, c) + state_of(j, c);
            bool ok = true;
            for (size_t di = 0; di < devs.size() && ok; ++di) ok = need <= room[di];
            if (ok) {
                k = c;
                break;
            }
        }
        if (k <= 0) {
            size_t need = 0;
            for (int j = 0; j < min_k; ++j) need += slot_of(j, min_k) + state_of(j, min_k);
            std::fprintf(stderr, "ds4 rowsplit: off (a group of %d layers needs %zu MB a device, the tightest has %zu)\n",
                         min_k, need >> 20, *std::min_element(room.begin(), room.end()) >> 20);
            return true;
        }
        rp.k = k;
        rp.ng = (nl + k - 1) / k;
        // the replicas: an arena a slot, its ring and compressor group once; no expert banks - every expert streams
        // through the device's ring, exactly as the CPU lane's experts do
        rp.slot_bytes.assign((size_t) k, 0);
        for (int j = 0; j < k; ++j) rp.slot_bytes[(size_t) j] = slot_of(j, k);
        rp.rep.assign(devs.size(), {});
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            rp.rep[di].resize((size_t) k);
            for (int j = 0; j < k; ++j) {
                RpRep& R = rp.rep[di][(size_t) j];
                if (!rp_arena(d, R, rp.slot_bytes[(size_t) j], err)) return false;
                int ratio = 0;
                for (int il = j; il < nl; il += k) {
                    const Layer& l = L[(size_t) il];
                    if (l.kv_src && l.ratio > 1) ratio = std::max(ratio, l.ratio);
                }
                if (!dalloc(d, R.l.win, (size_t) g.win * win_rb(), err, &d.caches)) return false;
                if (ratio > 0 && (!dalloc(d, R.c_kv, (size_t) ratio * g.hd, err, &d.caches) ||
                                  !dalloc(d, R.c_score, (size_t) ratio * g.hd, err, &d.caches)))
                    return false;
            }
        }
        // every expert a streamed one: the virtual slots 0 .. n_exp - 1, one map for the device (the replica running
        // uses it, and only one replica of a device ever runs at a time)
        rp.vs_pin.assign(devs.size(), nullptr);
        rp.vs_dev.assign(devs.size(), nullptr);
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            if (!halloc(rp.vs_pin[di], (size_t) g.n_exp, err) ||
                !dalloc(d, rp.vs_dev[di], (size_t) g.n_exp, err, &d.caches))
                return false;
            for (int e = 0; e < g.n_exp; ++e) rp.vs_pin[di][(size_t) e] = e;
            for (RpRep& R : rp.rep[di]) {
                R.l.vs_pin = rp.vs_pin[di];
                R.l.vs_dev = rp.vs_dev[di];
            }
        }
        // the wavefront's publications, host-mapped (the chunk that computed writes, the ones behind it read): the ring
        // and the compressor's group a (device, layer), the source rows a (device, source); and the pinned staging the
        // stage seeds and the prompt's source rows go through
        const size_t win_b = (size_t) g.win * win_rb();
        int max_chunk = 0;
        for (size_t di = 0; di < devs.size(); ++di) {
            const int grp = devs[di].grp;
            max_chunk = std::max(max_chunk, grp >= 0 && grp < (int) chunk_steps.size() ? chunk_steps[(size_t) grp] : chunk);
        }
        rp.ring_pub.assign(devs.size(), std::vector<uint8_t*>((size_t) nl, nullptr));
        rp.ring_flag.assign(devs.size(), std::vector<unsigned*>((size_t) nl, nullptr));
        rp.comp_pub.assign(devs.size(), std::vector<uint8_t*>((size_t) nl, nullptr));
        rp.comp_flag.assign(devs.size(), std::vector<unsigned*>((size_t) nl, nullptr));
        rp.src_pub.assign(devs.size(), std::vector<uint8_t*>(srcs.size(), nullptr));
        rp.src_flag.assign(devs.size(), std::vector<unsigned*>(srcs.size(), nullptr));
        rp.src_rows.assign(devs.size(), std::vector<int>(srcs.size(), 0));
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            for (int il = 0; il < nl; ++il) {
                if (!hmap(rp.ring_pub[di][(size_t) il], win_b, err) ||
                    !hmap(rp.ring_flag[di][(size_t) il], 8, err))
                    return false;
                const Layer& l = L[(size_t) il];
                if (!l.kv_src || l.ratio <= 1) continue;
                if (!hmap(rp.comp_pub[di][(size_t) il], 2 * (size_t) l.ratio * g.hd * 4, err) ||
                    !hmap(rp.comp_flag[di][(size_t) il], 8, err))
                    return false;
            }
            for (size_t si = 0; si < srcs.size(); ++si) {
                const int R = std::max(1, srcs[si].ratio);
                const int rows = std::max(kWin, max_chunk / R + 2);   // (a chunk's own rows: at most T / R + 1)
                rp.src_rows[di][si] = rows;
                if (!hmap(rp.src_pub[di][si], (size_t) rows * (ckv_rb() + ik_rb()), err) ||
                    !hmap(rp.src_flag[di][si], 8, err))
                    return false;
            }
        }
        rp.src_ik.assign(srcs.size(), 0);
        for (size_t si = 0; si < srcs.size(); ++si)
            rp.src_ik[si] = srcs[si].ik[(size_t) L[(size_t) srcs[si].layer].dev] != nullptr;
        rp.pulled.assign(devs.size(), std::vector<char>(srcs.size(), 0));
        rp.prog.reset(new std::atomic<int>[devs.size()]);
        for (size_t di = 0; di < devs.size(); ++di) rp.prog[di].store(-1);
        // the balance: a weight a device (mean 1, presettable with DS4_ROW_SPLIT_W by CUDA order), and the events its
        // layers are timed with; the learning reads each card's own layer time, never the wait for the chunk in front
        rp.row_w.assign(devs.size(), 1.0);
        rp.work_ms.assign(devs.size(), 0.0);
        rp.balance = rp_env_int("DS4_ROW_SPLIT_BALANCE", 1) != 0;
        rp.early_pub = rp_env_int("DS4_ROW_SPLIT_EARLY_PUB", 1) != 0;
        if (const char* ws = std::getenv("DS4_ROW_SPLIT_W")) {
            const char* p = ws;
            for (size_t di = 0; di < devs.size() && *p; ++di) {
                char* e = nullptr;
                const double v = std::strtod(p, &e);
                if (e == p) break;
                if (v > 0.0) rp.row_w[di] = std::min(4.0, std::max(0.25, v));
                p = *e == ',' ? e + 1 : e;
            }
        }
        rp.ev.assign(devs.size(), {});
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[(size_t) di];
            rp.ev[di].resize((size_t) (2 * rp.k), nullptr);
            cudaSetDevice(d.id);
            for (cudaEvent_t& e : rp.ev[di])
                if (cudaEventCreate(&e) != cudaSuccess) {
                    err = "the row split's balance events";
                    return false;
                }
        }
        for (auto& cp : chunks)
            if (cp != nullptr) cp->lay_override.assign(L.size(), nullptr);
        size_t mb = 0;
        for (size_t b : rp.slot_bytes) mb += b;
        std::fprintf(stderr,
                     "ds4 rowsplit: %d layers a group, %d groups, %.0f MB of dense weights a device (the expert slots "
                     "gave up that VRAM)\n",
                     rp.k, rp.ng, (double) mb / 1e6);
        rp.on = true;
        return true;
    }

    // one replica loaded for its layer: the layer's metadata from the canonical one, its banks emptied (the replica
    // owns no expert), its dense weights carved into the slot's arena; what the mode allocated once - the ring, the
    // compressor's group, the slot maps - is kept across the swaps
    bool load_replica(int di, int j, int il, std::string& err) {
        Dev& d = devs[(size_t) di];
        RpRep& R = rp.rep[(size_t) di][(size_t) j];
        Layer& src = L[(size_t) il];
        uint8_t* win = R.l.win;
        int* vs_pin = R.l.vs_pin;
        int* vs_dev = R.l.vs_dev;
        R.l = src;                        // the layout, the file's offsets, the sources, Engram, the name
        R.l.il = il;
        R.l.dev = di;
        R.l.grp = devs[(size_t) di].grp;
        R.l.nm = 1;
        R.l.mi = 0;
        R.l.mates.clear();
        R.l.ff0 = 0;
        R.l.ffn = g.ff;
        R.l.ah0 = 0;
        R.l.ahn = g.heads;
        R.l.og0 = 0;
        R.l.ogn = g.o_groups;
        R.l.mtp = false;
        R.l.host_list.resize((size_t) R.l.n_exp);
        for (int e = 0; e < R.l.n_exp; ++e) R.l.host_list[(size_t) e] = e;
        R.l.n_slots = 0;
        R.l.n_vram = 0;
        R.l.n_perm = 0;
        R.l.n_lend = 0;
        R.l.vram_base = nullptr;
        R.l.lend_base = nullptr;
        R.l.lend = false;
        R.l.lend_keep.clear();
        R.l.slot.clear();
        R.l.ram.clear();                  // (empty: flow_push reads every expert from the files)
        R.l.host.clear();
        R.l.hh.clear();
        R.l.heat.clear();
        R.l.cpu_hits.clear();
        R.l.tab_pin = nullptr;
        R.l.spare_vram = -1;
        R.l.spare_host = {nullptr, nullptr, nullptr, nullptr};
        R.l.snap_win = nullptr;
        R.l.snap_kv = nullptr;
        R.l.snap_sc = nullptr;
        R.l.ring_bak = nullptr;
        R.l.c_snap = nullptr;
        R.l.c_rows = nullptr;
        R.l.vram_tab = nullptr;
        R.l.swap_pending = false;
        R.l.vm.clear();
        R.l.win = win;
        R.l.c_kv = src.kv_src && src.ratio > 1 ? R.c_kv : nullptr;
        R.l.c_score = R.l.c_kv != nullptr ? R.c_score : nullptr;
        R.l.vs_pin = vs_pin;
        R.l.vs_dev = vs_dev;
        cudaSetDevice(d.id);
        R.bump.off = 0;
        d.bump = &R.bump;
        struct NoBump {   // (every way out: no stray dalloc lands in the arena)
            Dev& d;
            ~NoBump() { d.bump = nullptr; }
        } nb{d};
        if (!load_layer_dense(d, R.l, 1, 0, false, err, nullptr)) return false;
        return true;
    }
    // the stage's layers on this device, each in its slot (the layers it held for the stage before are overwritten)
    bool rp_load_group(int di, int stage, std::string& err) {
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            if (!load_replica(di, j, il, err)) return false;
        }
        return true;
    }

    // the rows a prompt needs for the mode (DS4_ROW_SPLIT_MIN, 8192; DS4_ROW_SPLIT_FORCE=1: any)
    int rp_min_rows() const {
        if (rp_env_int("DS4_ROW_SPLIT_FORCE", 0) != 0) return 1;
        return std::max(1, rp_env_int("DS4_ROW_SPLIT_MIN", 8192));
    }
    // a device's chunk cap: its prompt chunk's step
    int rp_cap(int di) const {
        const int grp = di >= 0 && (size_t) di < devs.size() ? devs[(size_t) di].grp : -1;
        return grp >= 0 && grp < (int) chunk_steps.size() ? chunk_steps[(size_t) grp] : chunk;
    }
    // the rows a round takes: every device's chunk cap, all in.  It comes from --prefill-chunk-max, so the mode fits
    // the prompt - a prompt of any size runs in rounds of these, the last one the remainder
    int64_t rp_round_rows() const {
        int64_t caps = 0;
        for (size_t di = 0; di < devs.size(); ++di) caps += rp_cap((int) di);
        return caps;
    }
    // a prompt's rounds: full ones, then the remainder.  A tail under one row a device takes rows from the round
    // before it (the last partition spreads over every card)
    std::vector<int64_t> rp_round_sizes(int n) const {
        std::vector<int64_t> out;
        const int nd = (int) devs.size();
        const int64_t caps = rp_round_rows();
        int64_t left = n;
        while (left > 0) {
            int64_t m = std::min(caps, left);
            left -= m;
            if (left > 0 && left < nd) {
                m -= (int64_t) nd - left;
                left = nd;
            }
            out.push_back(m);
        }
        return out;
    }
    // the wavefront's chunk order: every device but the head, then the head (its chunk carries the logits)
    void rp_order(std::vector<int>& order) const {
        order.clear();
        for (int di = 0; di < (int) devs.size(); ++di)
            if (di != head_dev) order.push_back(di);
        order.push_back(head_dev);
    }
    // a round's m rows on the chunks: proportional to their devices' weights, clamped by the caps, at least a row a
    // chunk.  Equal weights give the even split; a card the learning found faster takes more rows
    void rp_split_rows(int n, const std::vector<int>& order, std::vector<int>& T) const {
        const int nd = (int) order.size();
        std::vector<double> w((size_t) nd, 1.0), want((size_t) nd, 0.0);
        std::vector<int> cap((size_t) nd, 1);
        double wsum = 0;
        for (int f = 0; f < nd; ++f) {
            const int di = order[(size_t) f];
            w[(size_t) f] = di >= 0 && (size_t) di < rp.row_w.size() ? rp.row_w[(size_t) di] : 1.0;
            w[(size_t) f] = std::min(4.0, std::max(0.25, w[(size_t) f]));
            cap[(size_t) f] = std::max(1, rp_cap(di));
            wsum += w[(size_t) f];
        }
        T.assign((size_t) nd, 1);
        int64_t sum = nd;   // (a row a chunk first)
        for (int f = 0; f < nd; ++f) want[(size_t) f] = (double) n * w[(size_t) f] / wsum;
        for (int f = 0; f < nd; ++f) {
            int t = std::max(1, std::min(cap[(size_t) f], (int) std::floor(want[(size_t) f])));
            T[(size_t) f] = t;
            sum += t - 1;
        }
        while (sum < (int64_t) n) {   // the floors' rows: to the chunk furthest under its share that still has room
            int best = -1;
            for (int f = 0; f < nd; ++f)
                if (T[(size_t) f] < cap[(size_t) f] &&
                    (best < 0 || want[(size_t) f] - T[(size_t) f] > want[(size_t) best] - T[(size_t) best]))
                    best = f;
            if (best < 0) break;
            ++T[(size_t) best];
            ++sum;
        }
        while (sum > (int64_t) n) {   // (small rounds: the floor of one a chunk can overrun the share's tail)
            int best = -1;
            for (int f = 0; f < nd; ++f)
                if (T[(size_t) f] > 1 &&
                    (best < 0 || want[(size_t) f] - T[(size_t) f] < want[(size_t) best] - T[(size_t) best]))
                    best = f;
            if (best < 0) break;
            --T[(size_t) best];
            --sum;
        }
    }
    // the weights after a round: a card's own layer time a row of the share it carried, and how far that is from the
    // mean - a faster card earns a bigger share of the next round's rows, a slower one a smaller.  Half the step a
    // round (damped), held in [0.5, 2] and the mean at 1, so the even split is always the neutral point
    void rp_rebalance() {
        const size_t nd = devs.size();
        std::vector<double> c(nd, 0.0);
        double sum = 0.0;
        for (size_t di = 0; di < nd; ++di) {
            const int f = rp_run.chunk_of[di];
            if (f < 0) continue;
            const int T = rp_run.T[(size_t) f];
            if (T <= 0 || rp.work_ms[di] <= 0.0) continue;
            c[di] = rp.work_ms[di] / (double) T;   // ms a row this round
            sum += c[di];
        }
        std::fill(rp.work_ms.begin(), rp.work_ms.end(), 0.0);
        if (sum <= 0.0) return;
        const double mean = sum / (double) nd;
        std::vector<double> w(nd, 1.0);
        double wsum = 0.0;
        for (size_t di = 0; di < nd; ++di) {
            const double target = c[di] > 0.0 ? mean / c[di] : 1.0;
            double nw = 0.5 * rp.row_w[di] + 0.5 * target;
            nw = std::min(2.0, std::max(0.5, nw));
            w[di] = nw;
            wsum += nw;
        }
        std::string msg = "ds4 rowsplit balance: weights";
        char b[48];
        for (size_t di = 0; di < nd; ++di) {
            rp.row_w[di] = w[di] * (double) nd / wsum;
            std::snprintf(b, sizeof b, " CUDA%d %.2f", devs[di].id, rp.row_w[di]);
            msg += b;
        }
        std::fprintf(stderr, "%s (the next round's rows follow them)\n", msg.c_str());
    }
    // whether a prompt takes the mode: the plan is up, and the prompt is at least the floor and a row on every card
    // (any larger prompt runs in rounds)
    bool rp_ready(int n, int logits_rows, std::string& why) const {
        if (!rp.on) {
            why = "off";
            return false;
        }
        if (rp.k <= 0 || rp.ng <= 0 || rp.rep.size() != devs.size()) {
            why = "no plan";
            return false;
        }
        if (logits_rows > kWin) {
            why = "every row's logits";
            return false;
        }
        if (n < rp_min_rows()) {
            why = "rows under DS4_ROW_SPLIT_MIN";
            return false;
        }
        if (chunks.size() != devs.size()) {
            why = "no chunk buffers";
            return false;
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            const int cap = rp_cap((int) di);
            if (cap < 1) {
                why = "a device has no chunk cap";
                return false;
            }
            if (rp.rep[di].size() != (size_t) rp.k) {
                why = "the plan's slots";
                return false;
            }
        }
        if (n < (int) devs.size()) {
            why = "fewer rows than devices";
            return false;
        }
        if (rp_round_rows() < 2 * (int) devs.size()) {
            why = "the devices' chunk caps are too small for rounds";
            return false;
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            const Chunk& c = *chunks[di];
            if (!c.flow || c.ring == nullptr || c.stage == nullptr || c.S < 1 || c.ring_fill.empty() ||
                c.ring_free.empty() || c.lay_override.size() != L.size()) {
                why = "the flow is not up on every device";
                return false;
            }
            for (const Src& s : srcs)
                if ((size_t) di >= s.ckv_rp.size() || s.ckv_rp[di] == nullptr) {
                    why = "no row-split source copies";
                    return false;
                }
        }
        return true;
    }
    // the prompt's rows on the devices: one chunk each, the head device's last (its chunk carries the logits and the
    // drafter's taps); the chunks take rows by their devices' weights under their caps
    bool rp_partition(int n, int64_t start, std::string& err) {
        const int nd = (int) devs.size();
        RpRun& R = rp_run;
        rp_order(R.order);
        R.chunk_of.assign(devs.size(), -1);
        std::vector<int> cap((size_t) nd, 0);
        int64_t total = 0;
        for (int f = 0; f < nd; ++f) {
            cap[(size_t) f] = rp_cap(R.order[(size_t) f]);
            total += cap[(size_t) f];
        }
        for (int f = 0; f < nd; ++f)
            if (cap[(size_t) f] < 1) {
                err = "the row split needs a chunk cap on every device (CUDA" +
                      std::to_string(devs[(size_t) R.order[(size_t) f]].id) + " has none)";
                return false;
            }
        if (n < nd || (int64_t) n > total) {
            err = "internal: the row split was offered " + std::to_string(n) + " rows of " + std::to_string(total);
            return false;
        }
        rp_split_rows(n, R.order, R.T);
        R.p0.assign((size_t) nd, 0);
        R.pos0 = start;
        R.rows = 0;
        int64_t at = start;
        for (int f = 0; f < nd; ++f) {
            R.p0[(size_t) f] = at;
            at += R.T[(size_t) f];
            R.rows = std::max(R.rows, R.T[(size_t) f]);
            R.chunk_of[(size_t) R.order[(size_t) f]] = f;
        }
        R.n = n;
        return true;
    }
    // a copy between devices (a plain one on the same one), on the destination's stream: these cards peer only
    // through the driver's staging, and the source must sit still while it runs
    bool rp_copy(Dev& dst, const void* src, int src_id, void* dp, size_t bytes, cudaStream_t s, std::string& err) {
        if (bytes == 0 || dp == nullptr || src == nullptr) return true;
        const cudaError_t e = src_id == dst.id
                                  ? cudaMemcpyAsync(dp, src, bytes, cudaMemcpyDeviceToDevice, s)
                                  : cudaMemcpyPeerAsync(dp, dst.id, src, src_id, bytes, s);
        if (e == cudaSuccess) return true;
        err = "a row-split copy to CUDA" + std::to_string(dst.id) + " failed: " + cudaGetErrorString(e);
        return false;
    }
    // every device's source copies get the rows before the round (the canonical copies' [0, pos0 / ratio)): chunk 0's
    // layers read them from its first row.  Round 1's prefix is the decode cache's; round r's is what every round
    // before it harvested, so the canonical copies must be current (rp_src_harvest runs at every round's end)
    bool rp_prefix_seed(std::string& err) {
        bool ok = true;
        const int64_t pos0 = rp_run.pos0;
        for (size_t si = 0; si < srcs.size(); ++si) {
            Src& sr = srcs[si];
            const size_t holder = (size_t) L[(size_t) sr.layer].dev;
            const int R = std::max(1, sr.ratio);
            const int64_t r0 = pos0 / R;
            if (r0 <= 0) continue;
            if (r0 > (int64_t) sr.cap) {
                err = "internal: the row-split prefix runs past the source's rows";
                return false;
            }
            const uint8_t* ck = rp.ckv_can[si][holder];   // (the canonical marks: this prompt's are its rp copies)
            const uint8_t* ik = rp.ik_can[si][holder];
            for (size_t di = 0; di < devs.size(); ++di) {
                Dev& d = devs[di];
                cudaSetDevice(d.id);
                ok = rp_copy(d, ck, devs[holder].id, sr.ckv_rp[di], (size_t) r0 * ckv_rb(), d.st, err) && ok;
                if (rp.src_ik[si])
                    ok = rp_copy(d, ik, devs[holder].id, sr.ik_rp[di], (size_t) r0 * ik_rb(), d.st, err) &&
                         ok;
            }
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            cudaSetDevice(devs[di].id);
            CK(cudaStreamSynchronize(devs[di].st));
        }
        return ok;
    }
    // the prompt's own rows of every source, from each chunk's copy into every canonical one (the decode path keeps
    // those current row by row: after this run they must hold the new rows too)
    bool rp_src_harvest(std::string& err) {
        bool ok = true;
        sync_devs();
        for (size_t si = 0; si < srcs.size(); ++si) {
            const Src& sr = srcs[si];
            const int R = std::max(1, sr.ratio);
            for (int f = 0; f < (int) devs.size(); ++f) {
                const size_t cd = (size_t) rp_run.order[(size_t) f];
                const int64_t q0 = rp_run.p0[(size_t) f] / R;
                const int64_t q1 = (rp_run.p0[(size_t) f] + rp_run.T[(size_t) f]) / R;
                if (q1 <= q0) continue;
                const size_t nrow = (size_t) (q1 - q0);
                for (size_t di = 0; di < devs.size(); ++di) {
                    uint8_t* cc = rp.ckv_can[si][di];
                    uint8_t* ci = rp.ik_can[si][di];
                    if (cc == nullptr && ci == nullptr) continue;
                    Dev& d = devs[di];
                    cudaSetDevice(d.id);
                    if (cc != nullptr)
                        ok = rp_copy(d, sr.ckv_rp[cd] + (size_t) q0 * ckv_rb(), devs[cd].id,
                                     cc + (size_t) q0 * ckv_rb(), nrow * ckv_rb(), d.st, err) &&
                             ok;
                    if (ci != nullptr && rp.src_ik[si])
                        ok = rp_copy(d, sr.ik_rp[cd] + (size_t) q0 * ik_rb(), devs[cd].id,
                                     ci + (size_t) q0 * ik_rb(), nrow * ik_rb(), d.st, err) &&
                             ok;
                }
            }
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            cudaSetDevice(devs[di].id);
            CK(cudaStreamSynchronize(devs[di].st));
        }
        return ok;
    }
    // chunk 0's replicas before its layer: the canonical ring (and compressor group) at the prompt's position
    bool rp_stage_seed(int di, int stage, std::string& err) {
        bool ok = true;
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            const Layer& can = L[(size_t) il];
            RpRep& R = rp.rep[(size_t) di][(size_t) j];
            ok = rp_copy(d, can.win, devs[(size_t) can.dev].id, R.l.win, (size_t) g.win * win_rb(), d.st, err) && ok;
            if (R.l.c_kv != nullptr) {
                const size_t b = (size_t) R.l.ratio * g.hd * 4;
                ok = rp_copy(d, can.c_kv, devs[(size_t) can.dev].id, R.l.c_kv, b, d.st, err) && ok;
                ok = rp_copy(d, can.c_score, devs[(size_t) can.dev].id, R.l.c_score, b, d.st, err) && ok;
            }
        }
        return ok;
    }
    // the last chunk's replicas after its stage: the canonical ring (and compressor group) at the prompt's end (the
    // layers are used once, so the canonical state they were seeded from is still the prompt's start)
    bool rp_stage_harvest(int di, int stage, std::string& err) {
        bool ok = true;
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        CK(cudaStreamSynchronize(d.st));   // (the replicas' writes, before anything copies them out)
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            Layer& can = L[(size_t) il];
            RpRep& R = rp.rep[(size_t) di][(size_t) j];
            Dev& cd = devs[(size_t) can.dev];
            cudaSetDevice(cd.id);
            ok = rp_copy(cd, R.l.win, d.id, can.win, (size_t) g.win * win_rb(), cd.st, err) && ok;
            if (R.l.c_kv != nullptr) {
                const size_t b = (size_t) R.l.ratio * g.hd * 4;
                ok = rp_copy(cd, R.l.c_kv, d.id, can.c_kv, b, cd.st, err) && ok;
                ok = rp_copy(cd, R.l.c_score, d.id, can.c_score, b, cd.st, err) && ok;
            }
        }
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            Dev& cd = devs[(size_t) L[(size_t) il].dev];
            cudaSetDevice(cd.id);
            CK(cudaStreamSynchronize(cd.st));
        }
        cudaSetDevice(d.id);
        return ok;
    }
    // the prompt's sources: its own copies stand in for the canonical ones while it runs (layer_chunk reads and writes
    // srcs[...].ckv/ik where it runs), the canonical marks kept aside for rp_end
    void rp_begin_sources() {
        rp.ckv_can.assign(srcs.size(), {});
        rp.ik_can.assign(srcs.size(), {});
        for (size_t si = 0; si < srcs.size(); ++si) {
            Src& s = srcs[si];
            rp.ckv_can[si] = s.ckv;
            rp.ik_can[si] = s.ik;
            for (size_t di = 0; di < devs.size(); ++di) {
                s.ckv[di] = s.ckv_rp[di];
                s.ik[di] = s.ik_rp[di];
            }
        }
        rp.src_swapped = true;
    }
    void rp_end() {
        if (rp.src_swapped) {
            for (size_t si = 0; si < srcs.size() && si < rp.ckv_can.size(); ++si) {
                srcs[si].ckv = rp.ckv_can[si];
                srcs[si].ik = rp.ik_can[si];
            }
            rp.src_swapped = false;
        }
        for (auto& cp : chunks) {
            if (cp == nullptr) continue;
            cp->rp = false;
            std::fill(cp->lay_override.begin(), cp->lay_override.end(), nullptr);
        }
        for (auto& v : rp.pulled) std::fill(v.begin(), v.end(), 0);
    }

    // one device's worker for one stage: its own chunk through the stage's K layers - the wavefront's pulls before a
    // layer, its publications after - then (chunk 0) the stage's seed and (the last chunk) the canonical harvest, the
    // logits and the drafter's taps
    bool run_dev_rowsplit(int di, int stage, std::string& err) {
        const int f = rp_run.chunk_of[(size_t) di];
        if (f < 0) return true;
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nd = (int) devs.size(), D = g.dim, H = g.hc;
        const int64_t p0 = rp_run.p0[(size_t) f];
        const int T = rp_run.T[(size_t) f];
        const int fprev = f > 0 ? rp_run.order[(size_t) (f - 1)] : -1;
        const bool last_chunk = f + 1 == nd;
        // (temporary: the wavefront's hand-offs, named at the launch that faults; CUDA_LAUNCH_BLOCKING makes each
        // launch synchronous, so the error is that kernel's)
        auto rp_ck = [&](const char* what) -> bool {
            if (std::getenv("DS4_RP_CHECK") == nullptr) return true;
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                err = std::string("rowsplit ") + what + ": " + cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        // (temporary: a publication synced on the device at once, so a fault in it is named here and not at the next
        // layer's first launch)
        auto rp_ck_sync = [&](const char* what, int il) -> bool {
            if (std::getenv("DS4_RP_CHECK") == nullptr) return true;
            const cudaError_t e = cudaStreamSynchronize(s);
            if (e != cudaSuccess) {
                err = std::string("rowsplit ") + what + " layer " + std::to_string(il) + " dev " +
                      std::to_string(di) + ": " + cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        CK(cudaStreamSynchronize(s));
        CK(cudaStreamSynchronize(c.cs));
        // the stage's replicas in the slots, then the flow for one chunk of one stage
        if (!rp_load_group(di, stage, err)) return false;
        c.items.clear();
        c.base = c.n_read = c.n_dma = c.n_comp = 0;
        c.n_file = 0;
        std::fill(c.stage_item.begin(), c.stage_item.end(), -1);
        c.fl_layers.clear();
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            c.fl_layers.push_back(il);
        }
        c.fl_T.assign(1, T);
        c.spec_k = c.spec_il = -1;
        c.wait_ms = 0;
        c.it_file = c.it_ram = c.it_spec = c.it_spec_unused = 0;
        if (c.io != nullptr) {
            c.io_d0 = c.io->direct_bytes();
            c.io_b0 = c.io->buffered_bytes();
        }
        c.flow = true;
        c.rp = true;
        std::fill(c.lay_override.begin(), c.lay_override.end(), nullptr);
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            c.lay_override[(size_t) il] = &rp.rep[(size_t) di][(size_t) j].l;
        }
        if (c.fl_layers.empty()) return true;
        // the experts: every one of the stage's layers streams through this device's ring (no slots hold any), and the
        // flow reads and DMAs them as the layers route - the Streamer stays empty, as on the chunked path's flow
        Streamer S;
        flow_spec(c, 0, c.fl_layers[0], T);
        if (!flow_pump(c, err)) return false;
        // this stage's Engram rows (both chunks read their own positions)
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            const int eg = L[(size_t) il].eg;
            if (eg < 0) continue;
            const double te = now_ms();
            engram_rows(p0, T, eg, c.eg_host[0] + (size_t) eg * c.C * eg_per);
            if (d.mi == 0) st.engram_ms += now_ms() - te;
        }
        if (f == 0 && !rp_stage_seed(di, stage, err)) return false;
        if (stage == 0) {   // the prompt's own rows at layer 0
            for (int t = 0; t < T; ++t)
                std::memcpy(c.emb_host[0] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]), (size_t) D * 2);
            CK(cudaMemcpyAsync(c.emb, c.emb_host[0], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(c.emb_ev[0], s));
            embed(c.emb, T, c.h, D, H, s);
            init_pre(c.pre, T, s);
        }
        for (int j = 0; j < rp.k; ++j) {
            const int il = stage * rp.k + j;
            if (il >= g.nl) break;
            if (std::getenv("DS4_RP_CHECK") != nullptr) {   // (temporary: the fault before this layer, named)
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) {
                    err = "rowsplit at layer " + std::to_string(il) + " dev " + std::to_string(di) +
                          " (before it): " + cudaGetErrorString(e);
                    return false;
                }
            }
            RpRep& R = rp.rep[(size_t) di][(size_t) j];
            Layer& L0 = R.l;
            const unsigned sq = rp.seq + (unsigned) il + 1;
            // what the chunk before left at this layer: its ring, and the compressor's partial group
            int wait = 0;
            if (f > 0)
                wait = (int) (*(volatile const unsigned*) rp.ring_flag[(size_t) fprev][(size_t) il] - sq) < 0;
            live_line(di, 0, il, 0, 1, p0, T, "rowsplit", stage, rp.ng, wait, T, rp_run.round, rp_run.nround);
            if (f > 0) {
                Xfer x;
                x.src[0] = rp.ring_pub[(size_t) fprev][(size_t) il];
                x.dst[0] = L0.win;
                x.bytes[0] = (size_t) g.win * win_rb();
                x.n = 1;
                xfer_get(x, rp.ring_flag[(size_t) fprev][(size_t) il], sq, s);
                if (!rp_ck("xfer_get ring")) return false;
                if (L0.kv_src && L0.ratio > 1 && rp.comp_pub[(size_t) fprev][(size_t) il] != nullptr) {
                    const size_t b = (size_t) L0.ratio * g.hd * 4;
                    Xfer z;
                    z.src[0] = rp.comp_pub[(size_t) fprev][(size_t) il];
                    z.dst[0] = L0.c_kv;
                    z.bytes[0] = b;
                    z.src[1] = rp.comp_pub[(size_t) fprev][(size_t) il] + b;
                    z.dst[1] = L0.c_score;
                    z.bytes[1] = b;
                    z.n = 2;
                    xfer_get(z, rp.comp_flag[(size_t) fprev][(size_t) il], sq, s);
                    if (!rp_ck("xfer_get comp")) return false;
                }
            }
            // the sources' complete rows from every chunk before this one, once, before the first layer that reads them
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || rp.pulled[(size_t) di][(size_t) sidx]) continue;
                rp.pulled[(size_t) di][(size_t) sidx] = 1;
                if (f == 0) continue;
                const Src& sr = srcs[(size_t) sidx];
                const int Rs = std::max(1, sr.ratio);
                for (int de = 0; de < f; ++de) {
                    const int pdep = rp_run.order[(size_t) de];
                    const int64_t q0 = rp_run.p0[(size_t) de] / Rs;
                    const int64_t q1 = (rp_run.p0[(size_t) de] + rp_run.T[(size_t) de]) / Rs;
                    if (q1 <= q0) continue;   // (the writer's own range was empty: nothing published)
                    Xfer x;
                    x.src[0] = rp.src_pub[(size_t) pdep][(size_t) sidx];
                    x.dst[0] = sr.ckv_rp[(size_t) di] + (size_t) q0 * ckv_rb();
                    x.bytes[0] = (size_t) (q1 - q0) * ckv_rb();
                    x.n = 1;
                    if (rp.src_ik[(size_t) sidx] && sr.ik_rp[(size_t) di] != nullptr) {
                        x.src[1] = rp.src_pub[(size_t) pdep][(size_t) sidx] +
                                   (size_t) rp.src_rows[(size_t) pdep][(size_t) sidx] * ckv_rb();
                        x.dst[1] = sr.ik_rp[(size_t) di] + (size_t) q0 * ik_rb();
                        x.bytes[1] = (size_t) (q1 - q0) * ik_rb();
                        x.n = 2;
                    }
                    xfer_get(x, rp.src_flag[(size_t) pdep][(size_t) sidx], rp.seq + (unsigned) sidx + 1, s);
                    if (!rp_ck("xfer_get src")) return false;
                }
            }
            if (last_chunk && rp_run.final && g.n_mtp > 0) {   // the drafter's taps: this chunk's last rows, at its
                                                               // target layers (the prompt's own last round only)
                const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                if (it != g.d_targets.end()) {
                    const size_t jt = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                    const int nr = std::min(T, tap_cap);
                    hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                    CK(cudaStreamWaitEvent(s, ev_seed, 0));
                    CK(cudaMemcpy2DAsync(h_taps + jt * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                         (size_t) nr, cudaMemcpyDeviceToHost, s));
                }
            }
            WinPlan wp;
            wp.par = true;
            wp.async = true;
            wp.win = sq;
            wp.k = 0;
            wp.snap_t = -1;
            wp.eg_slot = 0;
            // (early publish) what the chunk behind this one waits on, handed to the layer so it can emit it at its
            // compressor instead of after the whole layer: the same bytes, roughly one layer earlier in the wavefront
            RpPub pub;
            const bool early = rp.early_pub && !last_chunk;
            if (early) {
                pub.ring.src[0] = L0.win;
                pub.ring.dst[0] = rp.ring_pub[(size_t) di][(size_t) il];
                pub.ring.bytes[0] = (size_t) g.win * win_rb();
                pub.ring.n = 1;
                pub.ring_flag = rp.ring_flag[(size_t) di][(size_t) il];
                pub.ring_seq = sq;
                if (L0.kv_src && L0.ratio > 1 && rp.comp_pub[(size_t) di][(size_t) il] != nullptr) {
                    const size_t b = (size_t) L0.ratio * g.hd * 4;
                    pub.comp.src[0] = L0.c_kv;
                    pub.comp.dst[0] = rp.comp_pub[(size_t) di][(size_t) il];
                    pub.comp.bytes[0] = b;
                    pub.comp.src[1] = L0.c_score;
                    pub.comp.dst[1] = rp.comp_pub[(size_t) di][(size_t) il] + b;
                    pub.comp.bytes[1] = b;
                    pub.comp.n = 2;
                    pub.comp_flag = rp.comp_flag[(size_t) di][(size_t) il];
                    pub.comp_seq = sq;
                }
                if (L0.kv_src && L0.src >= 0) {   // the completed rows, for the chunks behind
                    const int sidx = L0.src;
                    const int Rs = std::max(1, L0.ratio);
                    const int64_t g0 = p0 / Rs, g1 = (p0 + T) / Rs;
                    const int ng = (int) std::max<int64_t>(0, g1 - g0);
                    if (ng > 0) {
                        const Src& sr = srcs[(size_t) sidx];
                        pub.src.src[0] = sr.ckv_rp[(size_t) di] + (size_t) g0 * ckv_rb();
                        pub.src.dst[0] = rp.src_pub[(size_t) di][(size_t) sidx];
                        pub.src.bytes[0] = (size_t) ng * ckv_rb();
                        pub.src.n = 1;
                        if (rp.src_ik[(size_t) sidx] && sr.ik_rp[(size_t) di] != nullptr) {
                            pub.src.src[1] = sr.ik_rp[(size_t) di] + (size_t) g0 * ik_rb();
                            pub.src.dst[1] = rp.src_pub[(size_t) di][(size_t) sidx] +
                                             (size_t) rp.src_rows[(size_t) di][(size_t) sidx] * ckv_rb();
                            pub.src.bytes[1] = (size_t) ng * ik_rb();
                            pub.src.n = 2;
                        }
                        pub.src_flag = rp.src_flag[(size_t) di][(size_t) sidx];
                        pub.src_seq = rp.seq + (unsigned) sidx + 1;
                    }
                }
            }
            wp.rp_pub = early ? &pub : nullptr;
            // the balance's clock: from here to the end of its publications - this card's own layer, the wait for the
            // chunk in front of it (a stream wait before this record) excluded
            if (rp.balance) CK(cudaEventRecord(rp.ev[(size_t) di][2 * (size_t) j], s));
            if (!layer_chunk(L0, 0, p0, T, wp, S, err)) return false;
            if (std::getenv("DS4_RP_CHECK") != nullptr) {   // (temporary: the layer's own launches, synced and named)
                const cudaError_t e = cudaStreamSynchronize(s);
                if (e != cudaSuccess) {
                    err = "rowsplit layer " + std::to_string(il) + " dev " + std::to_string(di) + " T " +
                          std::to_string(T) + ": " + cudaGetErrorString(e);
                    return false;
                }
            }
            live_line(di, 0, il, 0, 1, p0, T, "rowsplit", stage, rp.ng, 0, T, rp_run.round, rp_run.nround);
            rp.prog[(size_t) di].store(il);
            if (last_chunk) {   // (no chunk behind it: its publications are none)
                if (rp.balance) CK(cudaEventRecord(rp.ev[(size_t) di][2 * (size_t) j + 1], s));
                continue;
            }
            // its publications: the ring and the compressor's group after this layer, and (at a writer) the source rows
            // (early publish: the layer emitted them at its compressor, so there is nothing left to send here)
            if (!early) {
                Xfer x;
                x.src[0] = L0.win;
                x.dst[0] = rp.ring_pub[(size_t) di][(size_t) il];
                x.bytes[0] = (size_t) g.win * win_rb();
                x.n = 1;
                xfer_put(x, rp.ring_flag[(size_t) di][(size_t) il], sq, d.xcnt, s);
                if (!rp_ck_sync("xfer_put ring", il)) return false;
                if (L0.kv_src && L0.ratio > 1 && rp.comp_pub[(size_t) di][(size_t) il] != nullptr) {
                    const size_t b = (size_t) L0.ratio * g.hd * 4;
                    Xfer z;
                    z.src[0] = L0.c_kv;
                    z.dst[0] = rp.comp_pub[(size_t) di][(size_t) il];
                    z.bytes[0] = b;
                    z.src[1] = L0.c_score;
                    z.dst[1] = rp.comp_pub[(size_t) di][(size_t) il] + b;
                    z.bytes[1] = b;
                    z.n = 2;
                    xfer_put(z, rp.comp_flag[(size_t) di][(size_t) il], sq, d.xcnt, s);
                    if (!rp_ck_sync("xfer_put comp", il)) return false;
                }
                if (L0.kv_src && L0.src >= 0) {   // the compressed rows (and index keys) it completed, for the chunks behind
                    const int sidx = L0.src;
                    const int Rs = std::max(1, L0.ratio);
                    const int64_t g0 = p0 / Rs, g1 = (p0 + T) / Rs;
                    const int ng = (int) std::max<int64_t>(0, g1 - g0);
                    if (ng > 0) {
                        const Src& sr = srcs[(size_t) sidx];
                        Xfer y;
                        y.src[0] = sr.ckv_rp[(size_t) di] + (size_t) g0 * ckv_rb();
                        y.dst[0] = rp.src_pub[(size_t) di][(size_t) sidx];
                        y.bytes[0] = (size_t) ng * ckv_rb();
                        y.n = 1;
                        if (rp.src_ik[(size_t) sidx] && sr.ik_rp[(size_t) di] != nullptr) {
                            y.src[1] = sr.ik_rp[(size_t) di] + (size_t) g0 * ik_rb();
                            y.dst[1] = rp.src_pub[(size_t) di][(size_t) sidx] +
                                       (size_t) rp.src_rows[(size_t) di][(size_t) sidx] * ckv_rb();
                            y.bytes[1] = (size_t) ng * ik_rb();
                            y.n = 2;
                        }
                        xfer_put(y, rp.src_flag[(size_t) di][(size_t) sidx], rp.seq + (unsigned) sidx + 1, d.xcnt, s);
                        if (!rp_ck_sync("xfer_put src", il)) return false;
                    }
                }
            }
            if (rp.balance) CK(cudaEventRecord(rp.ev[(size_t) di][2 * (size_t) j + 1], s));
        }
        if (last_chunk) {   // the last rows' logits (the final round's), from this chunk's state at the last layer
            if (rp_run.final && n_logits > 0) {
                const int t0 = T - n_logits;
                hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
            }
            if (!rp_stage_harvest(di, stage, err)) return false;
        }
        return true;
    }

    // a prompt through the row split: rounds of the devices' caps (a prompt that fits is one round), each one's
    // partition, its prefix, then every stage over every device's own chunk - the next round seeded from what this
    // one harvested
    bool forward_rowsplit(const int* tk, int n, int logits_rows, std::string& err) {
        if (n < 1) return true;
        std::string why;
        if (!rp_ready(n, logits_rows, why)) {
            err = "the row split cannot take this prompt (" + why + ")";
            return false;
        }
        if (pos + n > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < n; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        const int nd = (int) devs.size();
        const std::vector<int64_t> rounds = rp_round_sizes(n);
        logits_all = false;
        n_logits = 0;
        for (int t = 0; t < n; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        const double tl = now_ms();
        int plan_rows = 0;   // (the widest chunk any round gives: the arena the flow is described for)
        {
            std::vector<int> order, T;
            rp_order(order);
            for (int64_t m : rounds) {
                rp_split_rows((int) m, order, T);
                for (int t : T) plan_rows = std::max(plan_rows, t);
            }
        }
        if (!enter_prefill(plan_rows, err)) return false;
        prof_chunk = prof_on;
        live_base = pos;
        live_ntot = n;
        rp_begin_sources();   // (the prompt's copies stand in for the canonical ones; the prefix seeds read the marks)
        struct RpEnd {
            Impl* p;
            ~RpEnd() { p->rp_end(); }
        } rp_end_{this};
        int64_t done = 0;
        for (size_t ri = 0; ri < rounds.size(); ++ri) {
            const int64_t m = rounds[ri];
            if (!rp_partition((int) m, pos + done, err)) return false;
            rp_run.round = (int) ri + 1;
            rp_run.nround = (int) rounds.size();
            rp_run.final = ri + 1 == rounds.size();
            n_logits = rp_run.final ? std::min(logits_rows, rp_run.T[(size_t) (nd - 1)]) : 0;
            rp.seq = ++rp_prompt_seq * (unsigned) (g.nl + 1);   // (this round's flags never match an earlier one's)
            for (size_t di = 0; di < devs.size(); ++di) rp.prog[di].store(-1);
            for (auto& v : rp.pulled) std::fill(v.begin(), v.end(), 0);
            std::fill(rp.work_ms.begin(), rp.work_ms.end(), 0.0);   // (the balance's clock, this round)
            if (o.verbose || rounds.size() > 1) {
                std::string msg = "ds4 rowsplit round " + std::to_string(ri + 1) + "/" +
                                  std::to_string(rounds.size()) + " rows:";
                char rb[48];
                for (size_t di = 0; di < devs.size(); ++di) {
                    const int f = rp_run.chunk_of[di];
                    if (f < 0) continue;
                    std::snprintf(rb, sizeof rb, " CUDA%d %d", devs[di].id, rp_run.T[(size_t) f]);
                    msg += rb;
                }
                std::fprintf(stderr, "%s (a card's share follows its weight)\n", msg.c_str());
            }
            if (!rp_prefix_seed(err)) return false;   // (the canonical rows before the round: every chunk reads them)
            for (int stage = 0; stage < rp.ng; ++stage) {
                std::vector<std::string> errs(devs.size());
                for (size_t di = 0; di < devs.size(); ++di)
                    workers[di]->submit([this, di, stage, &errs] {
                        std::string e;
                        if (!run_dev_rowsplit((int) di, stage, e))
                            errs[di] = "dev " + std::to_string(di) + ": " + (e.empty() ? "a device's thread failed" : e);
                    });
                for (;;) {
                    bool all = true;
                    for (auto& w : workers) all = all && w->idle();
                    if (all) break;
                    if (progress) progress(done + m * (stage + 1) / rp.ng);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                for (const std::string& e : errs)
                    if (!e.empty()) {
                        if (std::getenv("DS4_RP_CHECK") != nullptr)   // (temporary: every device's own failure, not just the first)
                            std::fprintf(stderr, "ds4 rpfail: %s\n", e.c_str());
                        err = err.empty() ? e : err + " | " + e;
                    }
                if (!err.empty()) return false;
                // (the Monitor's wavefront: every card a finish of its group behind it, held until all of them are - its
                // row lands at the next group's first layer, hatched, so the strip shows the barrier, not a stale layer)
                if (stage + 1 < rp.ng && rp.k > 0)
                    for (size_t di = 0; di < devs.size(); ++di) {
                        const int f = rp_run.chunk_of[di];
                        if (f < 0) continue;
                        live_line((int) di, 0, (stage + 1) * rp.k, 0, 1, rp_run.p0[(size_t) f], rp_run.T[(size_t) f],
                                  "rowsplit", stage + 1, rp.ng, 1, rp_run.T[(size_t) f], rp_run.round, rp_run.nround);
                    }
                sync_devs();   // (the stage's devices all done: the next stage's seats are theirs)
                if (rp.balance)   // (each card's own layer time, the wait for the chunk in front excluded)
                    for (size_t d2 = 0; d2 < devs.size(); ++d2) {
                        Dev& dv = devs[d2];
                        cudaSetDevice(dv.id);
                        for (int j = 0; j < rp.k; ++j) {
                            const int il = stage * rp.k + j;
                            if (il >= g.nl) break;
                            float msv = 0.0f;
                            if (cudaEventElapsedTime(&msv, rp.ev[d2][2 * (size_t) j], rp.ev[d2][2 * (size_t) j + 1]) ==
                                cudaSuccess)
                                rp.work_ms[d2] += (double) msv;
                        }
                    }
            }
            if (!rp_src_harvest(err)) return false;   // (this round's rows into the canonical copies: the next prefix)
            if (rp.balance) rp_rebalance();           // (the weights the next round's shares follow)
            done += m;
        }
        st.layer_ms += now_ms() - tl;
        eg_stats();
        {
            Dev& dh = devs[(size_t) head_dev];
            cudaSetDevice(dh.id);
            const double th = now_ms();
            CK(cudaStreamSynchronize(dh.st));
            st.head_ms += now_ms() - th;
        }
        if (prof_chunk) {   // DS4_PROF: the run's GPU time by stage, every device's summed
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
            cudaSetDevice(devs[(size_t) head_dev].id);
            prof_collect();
            std::fprintf(stderr,
                         "ds4 prompt GPU time, %d tokens: attention %.0f ms, routing + shared expert %.0f ms, experts "
                         "%.0f ms (with the waits for streamed ones), between layers %.0f ms\n",
                         n, cprof_ms[1], cprof_ms[2], cprof_ms[3], cprof_ms[0]);
            for (double& v : cprof_ms) v = 0;
            prof_chunk = false;
        }
        if (o.verbose || std::getenv("DS4_FLOW_STATS") != nullptr) {
            const double ms = now_ms() - tl;
            if (rounds.size() > 1)
                std::fprintf(stderr, "ds4 prompt rounds, %d tokens: %zu rounds of up to %lld rows (a round all the devices' "
                                     "chunk caps), %d rows a device a round at most\n",
                             n, rounds.size(), (long long) rp_round_rows(), plan_rows);
            for (size_t di = 0; di < chunks.size(); ++di) {
                const Chunk& c = *chunks[di];
                if (!c.flow) continue;
                const double db = (double) (c.io->direct_bytes() - c.io_d0) / 1e9;
                const double bb = (double) (c.io->buffered_bytes() - c.io_b0) / 1e9;
                const double fb = db + bb;
                std::fprintf(stderr,
                             "ds4 prompt stream, CUDA%d: %d tokens of %d in %.0f ms, %d of %d layers a group; %llu "
                             "experts streamed: %llu from the files (%.1f GB, %.2f GB/s), %llu from RAM; the compute "
                             "waited %.0f ms for reads\n",
                             devs[di].id, n, plan_rows, ms, rp.k, rp.ng,
                             (unsigned long long) (c.it_file + c.it_ram + c.it_lc), (unsigned long long) c.it_file, fb,
                             fb / std::max(1e-3, ms / 1000.0), (unsigned long long) c.it_ram, c.wait_ms);
            }
        }
        if (pm.on) {   // (the picks' spread for pick_path)
            double picked = 0;
            for (const auto& cp : chunks)
                if (cp != nullptr && cp->flow) picked += (double) (cp->it_file + cp->it_ram) - (double) cp->it_spec_unused;
            pm_learn(n, picked);
        }
        pos += n;
        st.tokens += n;
        st.path = 2;   // (the Monitor's Prefill tab: row-split)
        if (!exit_prefill(err)) return false;
        cache_tick(n);
        const int nr = std::min(rp_run.T[(size_t) (nd - 1)], tap_cap);
        if (!seed_drafter(pos - nr, nr, 0, err)) return false;
        return true;
    }

    // ---- a prompt's path: the flow or windows of 16
    // this PC's speeds for pick_path (every device on the flow, the experts in place): the DMA from pinned staging to a
    // GPU; the readers copying RAM into staging; the disk read by them (experts not in RAM); the CPU lane on experts in
    // RAM and on ones it reads from the disk first - each timed once, a few experts' worth (DS4_PATH_PICK=0: always the
    // flow, as before)
    void calibrate_paths() {
        if (chunk <= 0 || chunk_min != kFlowChunkMin || std::getenv("DS4_CHUNK_MIN") != nullptr) return;
        if (const char* v = std::getenv("DS4_PATH_PICK"))
            if (std::atoi(v) == 0) return;
        int di = -1;
        for (size_t i = 0; i < chunks.size() && di < 0; ++i)
            if (chunks[i] != nullptr && chunks[i]->flow && chunks[i]->S >= 2) di = (int) i;
        if (di < 0) return;
        Chunk& c = *chunks[(size_t) di];
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const double t0 = now_ms();
        // the DMA: staging -> the ring (once to warm up, once timed)
        const size_t nb = std::min({(size_t) 256 << 20, (size_t) c.S * c.stage_stride, (size_t) c.ring_n * c.ring_stride});
        double ms = 0;
        for (int rep = 0; rep < 2; ++rep) {
            const double a = now_ms();
            if (cudaMemcpyAsync(c.ring, c.stage, nb, cudaMemcpyHostToDevice, c.cs) != cudaSuccess ||
                cudaStreamSynchronize(c.cs) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            ms = now_ms() - a;
        }
        pm.dma = (double) nb / std::max(1e-3, ms / 1000.0);
        // the readers: RAM -> staging (a buffer of its own, touched first), in as many jobs as they are
        auto run_jobs = [&](std::vector<IoQueue::Job>& jobs, std::atomic<int>& left) {
            const double a = now_ms();
            left.store((int) jobs.size());
            c.io->push(jobs.data(), (int) jobs.size());
            while (left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            return now_ms() - a;
        };
        {
            const size_t part = std::min(nb, (size_t) 64 << 20) / 8 / 4096 * 4096;
            std::vector<uint8_t> src(part * 8, 1);
            std::vector<IoQueue::Job> jobs(8);
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < 8; ++i) {
                jobs[(size_t) i].mem = src.data() + (size_t) i * part;
                jobs[(size_t) i].dst = c.stage + (size_t) i * part;
                jobs[(size_t) i].len = part;
                jobs[(size_t) i].left = &left;
                jobs[(size_t) i].bad = &bad;
            }
            run_jobs(jobs, left);
            const double t1 = run_jobs(jobs, left), t2 = run_jobs(jobs, left);
            pm.copy = (double) (part * 8) / std::max(1e-3, std::min(t1, t2) / 1000.0);
        }
        // experts not in RAM (none of their sampled pages resident), of the trunk layer with the most CPU-lane ones:
        // up to min(8, staging slots) read by the readers as the flow reads them, then 8 others through the CPU lane
        // three times (from the disk, then in RAM: the faster of two)
        int best = -1, most = 0;
        for (int il = 0; il < g.nl; ++il) {
            int n = 0;
            for (int e = 0; e < L[(size_t) il].n_exp; ++e) n += L[(size_t) il].slot[(size_t) e] < 0;
            if (n > most) most = n, best = il;
        }
        if (best < 0) return;
        const Layer& l = L[(size_t) best];
        std::vector<int> cold;
        const int n_read = std::min(8, c.S), n_cpu = 8;
        for (int e = 0; e < l.n_exp && (int) cold.size() < n_read + n_cpu; ++e) {
            if (l.slot[(size_t) e] >= 0 || (!l.ram.empty() && l.ram[(size_t) e] >= 0)) continue;
            bool res = false;
            for (int r = 0; r < 3; ++r) res = res || os::resident_sampled(mx_mat(l, e, r), mx_part_bytes(l, r), 4);
            if (!res) cold.push_back(e);
        }
        if ((int) cold.size() >= n_read + n_cpu) {
            std::vector<IoQueue::Job> jobs;
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < n_read; ++i) {
                uint8_t* sb = c.stage + (size_t) i * c.stage_stride;
                for (int r = 0; r < 3; ++r) {
                    IoQueue::Job j;
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) cold[(size_t) i] * l.mx_stride[r];
                    j.len = mx_part_bytes(l, r);
                    j.dst = sb;
                    j.left = &left;
                    j.bad = &bad;
                    j.map = pk.at(l.mx_file[r], j.off);
                    jobs.push_back(j);
                    sb += flow_region(r);
                }
            }
            const double rd = run_jobs(jobs, left);
            if (!bad.load()) pm.disk = (double) n_read * (double) l.blob / std::max(1e-3, rd / 1000.0);
        }
        {   // the CPU lane: entries of one token (as most of a window's are), from the disk (if any were cold), then RAM
            const size_t first = (int) cold.size() >= n_read + n_cpu ? (size_t) n_read : 0;
            std::vector<Ent> ents;
            for (int i = 0; i < n_cpu; ++i) {
                Ent en;
                en.e = first + (size_t) i < cold.size() ? cold[first + (size_t) i] : -1;
                if (en.e < 0)   // (no cold ones: the first CPU-lane experts, wherever they are)
                    for (int e = 0, k = 0; e < l.n_exp && en.e < 0; ++e)
                        if (l.slot[(size_t) e] < 0 && k++ == i) en.e = e;
                if (en.e < 0) break;
                en.n = 1;
                en.slot0 = i;
                en.tok[0] = 0;
                ents.push_back(en);
            }
            if (ents.empty()) return;
            std::vector<float> x((size_t) g.dim);
            for (size_t i = 0; i < x.size(); ++i) x[i] = 0.02f * std::sin(0.37f * (float) i);
            const double bytes = (double) ents.size() * (double) l.blob;
            double a = now_ms();
            cpu_experts(l, x.data(), ents, (size_t) g.dim);
            const double t_first = now_ms() - a;
            double t_warm = 1e30;
            for (int rep = 0; rep < 2; ++rep) {
                a = now_ms();
                cpu_experts(l, x.data(), ents, (size_t) g.dim);
                t_warm = std::min(t_warm, now_ms() - a);
            }
            pm.cpu = bytes / std::max(1e-3, t_warm / 1000.0);
            if (first > 0 || (int) cold.size() >= n_cpu) pm.cpu_cold = bytes / std::max(1e-3, t_first / 1000.0);
        }
        pm.on = pm.dma > 0 && pm.copy > 0 && pm.cpu > 0;
        if (o.verbose) {
            char dk[32] = "not measured", cc[32] = "not measured";
            if (pm.disk > 0) std::snprintf(dk, sizeof dk, "%.2f GB/s", pm.disk / 1e9);
            if (pm.cpu_cold > 0) std::snprintf(cc, sizeof cc, "%.2f GB/s", pm.cpu_cold / 1e9);
            std::fprintf(stderr,
                         "ds4: prompts through the flow or windows by this PC's speeds: DMA %.1f GB/s, RAM to staging "
                         "%.1f GB/s, the disk %s; the CPU lane %.1f GB/s in RAM, %s from the disk (%.0f ms)\n",
                         pm.dma / 1e9, pm.copy / 1e9, dk, pm.cpu / 1e9, cc, now_ms() - t0);
        }
    }
    // layer l's CPU-lane experts (VRAM holds the rest) and the chance a token picks a given one of them: the profile's
    // share of the layer's picks they took, spread evenly over them, times pm.m (what the flow's runs found)
    double pm_rate(const Layer& l, int& n_cold) const {
        const uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        double sum = 0, cold = 0;
        n_cold = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            sum += pr[e];
            if (l.slot[(size_t) e] < 0) cold += pr[e], ++n_cold;
        }
        if (n_cold == 0) return 0.0;
        const double share = sum > 0 && cold > 0 ? cold / sum : (double) n_cold / l.n_exp;
        return pm.m * (double) l.k_exp * share / n_cold;
    }
    // the CPU-lane experts n tokens pick in the trunk's layers, expected (each layer's n_cold (1 - e^(-rate n)))
    double pm_picks(int n) const {
        double d = 0;
        for (int il = 0; il < g.nl; ++il) {
            int nc = 0;
            const double a = pm_rate(L[(size_t) il], nc);
            d += nc * (1.0 - std::exp(-a * n));
        }
        return d;
    }
    // a run of one chunk of n tokens on the flow picked `picked` CPU-lane experts: pm.m to what makes the expectation
    // that (bisection over its log; a running mean of the log)
    void pm_learn(int n, double picked) {
        if (!pm.on || n < 2 || picked < 1) return;
        const double m0 = pm.m;
        double lo = std::log(1e-3), hi = std::log(1e3);
        for (int it = 0; it < 40; ++it) {
            pm.m = std::exp(0.5 * (lo + hi));
            (pm_picks(n) < picked ? lo : hi) = 0.5 * (lo + hi);
        }
        const double obs = 0.5 * (lo + hi);
        pm.m = std::exp(pm.n_m == 0 ? obs : 0.7 * std::log(m0) + 0.3 * obs);
        ++pm.n_m;
    }
    // a prompt of n tokens that the flow could take (n >= chunk_min): true the flow, false windows of 16 - each path's
    // time predicted from the bytes of CPU-lane experts it would move (the flow: each picked one once, n_cold (1 -
    // e^(-rate n)) a layer, pm_rate) or compute (windows: each window's own picks), those in RAM (each layer's share:
    // sampled pages of up to 32 of its CPU-lane experts, mincore / QueryWorkingSetEx; the RAM tier's) and those not at
    // this PC's speeds - the flow's times what it really took against its predictions so far for prompts of that size
    // (pm_k), windows' once some ran what a window took times the windows; the flow unless windows are predicted
    // faster (by half, with less than kPickRes in RAM); said with --verbose
    bool pick_path(int n) {
        pm.last = -1;
        if (!pm.on || n >= kPickMax) {
            pm.last = 1;   // (the flow: the Monitor's Prefill tab says which path this prompt takes)
            st.path = 1;
            return true;
        }
        double f_res = 0, f_cold = 0, w = 0;
        const int nw = n / kWin, rem = n % kWin;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            int nc = 0;
            const double a = pm_rate(l, nc);
            if (nc == 0) continue;
            int sampled = 0, res = 0, seen = 0;
            const int step = std::max(1, nc / 32);
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0 || seen++ % step != 0) continue;
                bool r = !l.ram.empty() && l.ram[(size_t) e] >= 0;
                if (!r) {
                    r = true;
                    for (int p = 0; p < 3 && r; ++p) r = os::resident_sampled(mx_mat(l, e, p), mx_part_bytes(l, p), 2);
                }
                ++sampled;
                res += r;
            }
            const double share = sampled > 0 ? (double) res / sampled : 0.0, b = (double) l.blob;
            auto u = [&](int t) { return nc * (1.0 - std::exp(-a * t)); };
            f_res += share * u(n) * b;
            f_cold += (1.0 - share) * u(n) * b;
            w += ((double) nw * u(kWin) + (rem > 0 ? u(rem) : 0.0)) * b;
        }
        const double r_fr = std::min(pm.copy, pm.dma), r_fc = pm.disk > 0 ? std::min(pm.disk, pm.dma) : r_fr;
        const double r_wc = pm.cpu_cold > 0 ? pm.cpu_cold : 1.0 / (1.0 / r_fc + 1.0 / pm.cpu);
        const double t_flow = f_res / r_fr + f_cold / r_fc;
        const double t_win = std::max(0.0, w - f_cold) / pm.cpu + f_cold / r_wc;
        const int cls = pm_cls(n), n_win = (n + kWin - 1) / kWin;
        // (windows: once some ran, what a window took on average times the windows the prompt needs - each pays the
        // fixed costs of every layer on every GPU again, which a pass of the flow pays once)
        const double pf = pm_k(cls) * t_flow, pw = pm.win_n > 0 ? pm.win_s * n_win : t_win;
        // (the flow unless windows are predicted faster - in half its time at least, unless kPickRes of the CPU-lane
        // experts' bytes are in RAM: the flow wins on every PC whose experts come from an SSD, where a wrong pick would
        // cost the most; windows win where they are all in RAM)
        const double res_share = f_res / std::max(1.0, f_res + f_cold);
        const bool flow = !(pw < pf && (res_share >= kPickRes || 2.0 * pw <= pf));
        pm.last = flow ? 1 : 0;
        st.path = pm.last;
        pm.last_cls = cls;
        pm.last_nw = n_win;
        pm.last_pred = flow ? t_flow : t_win;
        if (o.verbose)
            std::fprintf(stderr,
                         "ds4: a prompt of %d tokens through %s: predicted %.2f s, %s %.2f s (%.0f%% of its CPU-lane "
                         "experts' %.1f GB in RAM)\n",
                         n, flow ? "the flow" : "windows", flow ? pf : pw, flow ? "windows" : "the flow", flow ? pw : pf,
                         100.0 * res_share, (f_res + f_cold) / 1e9);
        return flow;
    }
    // a prompt's size class for pm.k: 2-31, 32-127, 128-511, 512 tokens and more
    static int pm_cls(int n) { return n < 32 ? 0 : n < 128 ? 1 : n < 512 ? 2 : 3; }
    // the flow's time factor for size class c: its own, else its nearest class's, else 1 (never the windows': theirs
    // is mostly the fixed costs a window pays again and again)
    double pm_k(int c) const {
        for (int dc = 0; dc < 4; ++dc)
            for (int cc : {c - dc, c + dc})
                if (cc >= 0 && cc < 4 && pm.kn[0][cc] > 0) return pm.k[0][cc];
        return 1.0;
    }
    // n tokens through windows of 16, the last one's logits_rows rows of logits
    bool feed_windows(const int* tokens, int n, int logits_rows, std::string& err) {
        logits_all = false;
        for (int a = 0; a < n; a += kWin) {
            const int T = std::min(kWin, n - a);
            if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, err)) return false;
        }
        return true;
    }
    // the picked path took ms: the flow's predictions for the prompt's size class scaled by what it really takes (taken
    // / predicted), windows' time a window - running means
    void path_done(double ms) {
        if (pm.last < 0 || pm.last_pred <= 0) return;
        st.path = pm.last;   // (the Monitor's Prefill tab: which path this prompt took)
        if (pm.last == 0) {
            const double w = ms / 1000.0 / std::max(1, pm.last_nw);
            pm.win_s = pm.win_n == 0 ? w : 0.7 * pm.win_s + 0.3 * w;
            ++pm.win_n;
        } else {
            const double ratio = std::min(8.0, std::max(0.25, ms / 1000.0 / pm.last_pred));
            double& k = pm.k[0][pm.last_cls];
            int64_t& n = pm.kn[0][pm.last_cls];
            k = n == 0 ? ratio : 0.7 * k + 0.3 * ratio;
            ++n;
        }
        pm.last = -1;
    }
    // a run layer-major? - one device holding every layer on the flow, several chunks, and what the chunks after the
    // first would read from the files again more than the chunks' states moved to host memory and back at every layer
    // (DS4_LAYER_MAJOR=0: never); then its host memory: the chunks' states (pinned, else pageable) and the layer cache,
    // the largest layer's experts read from the files if RAM allows (ram_avail() less kRamMargin; DS4_LAYER_CACHE_MB
    // sets it) - at least one expert, or not layer-major
    bool lm_begin(const Run& run) {
        static const char* env = std::getenv("DS4_LAYER_MAJOR");
        if (devs.size() != 1 || run.T.size() < 2 || (env != nullptr && std::atoi(env) == 0)) return false;
        Chunk& c = *chunks[0];
        if (!c.flow) return false;
        size_t need = 0, n_exp = 0, blob_max = 0;
        double file_b = 0;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            size_t ne = 0;
            for (int e : l.host_list)
                if (l.ram.empty() || l.ram[(size_t) e] < 0) ++ne;
            const size_t stride = (l.blob + 63) & ~(size_t) 63;
            need = std::max(need, ne * stride);
            n_exp = std::max(n_exp, (size_t) l.n_exp);
            blob_max = std::max(blob_max, stride);
            file_b += (double) ne * (double) l.blob;
        }
        int64_t rows = 0;
        for (int T : run.T) rows += T;
        const size_t hb = chunk_hand_bytes(c.C), state_b = run.T.size() * hb;
        const double moved = 2.0 * (double) rows * ((double) hb / (double) chunk) * (double) g.nl;
        if (need == 0 || (double) (run.T.size() - 1) * file_b <= moved) return false;
        size_t lc_bytes = need;
        if (const char* mb = std::getenv("DS4_LAYER_CACHE_MB")) lc_bytes = std::min(need, (size_t) std::atoll(mb) << 20);
        else {
            const uint64_t avail = ram_avail();
            if (avail != ~uint64_t{0})
                lc_bytes = std::min<uint64_t>(need, avail > kRamMargin + state_b ? avail - kRamMargin - state_b : 0);
        }
        if (lc_bytes < blob_max) return false;
        void* st_p = nullptr;
        c.lm_pinned = cudaHostAlloc(&st_p, state_b, cudaHostAllocPortable) == cudaSuccess;
        if (!c.lm_pinned) {
            cudaGetLastError();
            st_p = os::anon_alloc(state_b, -1, 1);
            if (st_p == nullptr) return false;
        }
        c.lc = (uint8_t*) os::anon_alloc(lc_bytes, -1, 1);
        if (c.lc == nullptr) {
            if (c.lm_pinned) cudaFreeHost(st_p);
            else os::anon_free(st_p, state_b);
            return false;
        }
        c.lm_state = (uint8_t*) st_p;
        c.lm_state_bytes = state_b;
        c.lc_bytes = lc_bytes;
        c.lc_of.assign(n_exp, -1);
        c.lc_writer.assign(n_exp, -1);
        c.lm = true;
        lm_layers.store(0);
        return true;
    }
    void lm_end() {
        for (size_t di = 0; di < chunks.size(); ++di) {
            Chunk& c = *chunks[di];
            if (!c.lm) continue;
            cudaSetDevice(devs[di].id);
            cudaStreamSynchronize(devs[di].st);   // (nothing of the run's still reads or writes them: the device's
            cudaStreamSynchronize(c.cs);          // copies done, every read the run issued done - a failed run's too)
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            if (c.lm_pinned) cudaFreeHost(c.lm_state);
            else os::anon_free(c.lm_state, c.lm_state_bytes);
            os::anon_free(c.lc, c.lc_bytes);
            c.lm_state = c.lc = nullptr;
            c.lm_state_bytes = c.lc_bytes = 0;
            c.lm = c.lm_pinned = false;
        }
    }

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot = -1) {
        if (n < 1) return true;
        if (pos + n > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < n; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        // each group's run of chunks: group 0 divides the prompt as the flat path always has; a later group's chunks are
        // whole chunks of the group before it (m of them, its step / that one's), so every consumer chunk is a union of
        // whole feeder chunks - which is what lets the hand-offs and the cross-group KV pulls take a piece at a time
        std::vector<Run> runs(groups.size());
        {
            const int st = std::max(1, chunk_steps[0]);   // (the feeder's step: at most the global chunk)
            const int nc = (n + st - 1) / st, per = (n + nc - 1) / nc;
            Run& r = runs[0];
            for (int a = 0; a < n; a += per) {
                r.p0.push_back(pos + a);
                r.T.push_back(std::min(per, n - a));
                r.seq.push_back(++win_seq);
            }
        }
        for (size_t gi = 1; gi < runs.size(); ++gi) {
            const int m = std::max(1, chunk_steps[gi] / std::max(1, chunk_steps[gi - 1]));
            const Run& up = runs[gi - 1];
            Run& r = runs[gi];
            for (size_t j = 0; j < up.T.size(); j += (size_t) m) {
                const size_t j1 = std::min(up.T.size(), j + (size_t) m);
                const int64_t e = up.p0[j1 - 1] + (int64_t) up.T[j1 - 1];
                r.p0.push_back(up.p0[j]);
                r.T.push_back((int) (e - up.p0[j]));
                r.seq.push_back(++win_seq);
            }
        }
        // the decoder tail (DS4_TAIL_PREFILL): the layers after the candidate source layer run a second pass over the
        // last rows only.  The source layer's last chunk must hold the whole tail (its saved h | pre | topk | cand);
        // the runs below still tile the whole prompt for the first pass.
        const bool tail_on = tail_feed(runs, n, logits_rows);
        std::vector<Run> truns;
        if (tail_on) {
            truns.resize(groups.size());
            const int64_t t0 = pos + n - tail_rows();
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                bool dec = false;
                for (int il = tail_first(); il < g.nl; ++il) dec = dec || L[(size_t) il].grp == (int) gi;
                if (!dec) continue;
                Run& r = truns[gi];
                r.p0.push_back(t0);
                r.T.push_back(tail_rows());
                r.seq.push_back(++win_seq);
                r.stage = 1;
                if (snap_slot >= 0) r.snap_pos = pos + n - 1;
            }
        }
        int prog0 = head_dev;
        if (tail_on) {   // (the head's group is decoder-only: the progress follows the last encoder layer's device)
            prog0 = -1;
            for (int il = 0; il <= g.cand_src && il < g.nl; ++il) prog0 = L[(size_t) il].dev;
        }
        const Run& prun = runs[(size_t) devs[(size_t) prog0].grp];   // the first pass's progress
        const Run& frun = tail_on ? truns[(size_t) devs[(size_t) head_dev].grp]   // T_last, n_logits, the taps
                                  : runs[(size_t) devs[(size_t) head_dev].grp];
        for (const Run& r : runs) {   // each run tiles [pos, pos + n) exactly, every chunk within its group's step
            int64_t at = pos;
            bool ok = !r.T.empty();
            for (size_t k = 0; k < r.T.size() && ok; ++k) {
                ok = r.T[k] > 0 && r.p0[k] == at;
                at += r.T[k];
            }
            if (!ok || at != pos + n) {
                err = "internal: the prompt's chunks do not tile it";
                return false;
            }
        }
        logits_all = logits_rows > kWin;
        if (logits_all) chunk_logits.resize((size_t) n * g.vocab);
        for (int t = 0; t < n; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        // the run's widest chunk: what the buffers are used to, so the prompt takes back only that much of the arena
        int64_t run_rows = 0;
        for (const Run& r : runs)
            for (int t : r.T) run_rows = std::max<int64_t>(run_rows, t);
        for (const Run& r : truns)
            for (int t : r.T) run_rows = std::max<int64_t>(run_rows, t);
        if (!enter_prefill(run_rows, err)) return false;   // (the arena's experts out, its buffers zeroed)
        // expert-parallel prefill: an encoder pass of one chunk a group, every layer's group a single member, no
        // decoder tail (its partners would wait on a layer no owner runs)
        ep_run = false;
        if (ep_alloc && ep_do() && !tail_on) {
            bool ok = true;
            for (const Run& r : runs) ok = ok && r.T.size() == 1;
            for (const Layer& l : L) ok = ok && (l.mtp || l.nm == 1);
            if (ok) {
                ep_base = ++ep_n * 256u;   // (a prompt's flags apart from every other's)
                ep_run = true;
            } else if (o.verbose) {
                std::fprintf(stderr, "ds4 ep: off (not a single-chunk pass over plain layers)\n");
            }
        }
        struct EpEnd {   // (every way out: the flag off before the next pass)
            Impl* p;
            ~EpEnd() { p->ep_run = false; }
        } ep_end_{this};
        // DS4_PASS_LEND: pass 1's borrow - the decoder tail's banks to the encoder layers (pl_borrow); the plan below is
        // the borrowed one, and every way out of the prompt puts the decoder's addressing back before exit_prefill
        struct PlEnd {
            Impl* p;
            bool on = false;
            ~PlEnd() {
                if (!on) return;
                std::string e2;
                p->pl_restore(e2);
            }
        } pl_end_{this};
        if (pl_do() && tail_on && !ep_run && !pl_pairs.empty()) {
            pl_end_.on = true;
            if (!pl_borrow(0, err)) return false;
        }
        plan_streams();
        if (snap_slot >= 0) {   // feed_mark: the state before the run's last row, kept by its layers on the way
            for (Run& r : runs) r.snap_pos = pos + n - 1;
            for (int il = 0; il < g.nl; ++il) {
                Layer& l = L[(size_t) il];
                Dev& d = devs[(size_t) l.dev];
                if (l.snap_win == nullptr && !dalloc(d, l.snap_win, (size_t) g.win * win_rb(), err)) return false;
                if (l.c_kv != nullptr && l.snap_kv == nullptr &&
                    (!dalloc(d, l.snap_kv, (size_t) l.ratio * g.hd, err) || !dalloc(d, l.snap_sc, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
        }
        const int T_last = frun.T.back();
        n_logits = logits_all ? 0 : std::min(logits_rows, T_last);
        const double tl = now_ms();
        {   // each chunk's end on the progress device, for the progress reports
            cudaSetDevice(devs[(size_t) prog0].id);
            while (chunk_done.size() < prun.T.size()) {
                cudaEvent_t e = nullptr;
                CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                chunk_done.push_back(e);
            }
            chunk_rec.reset(new std::atomic<int>[chunk_done.size()]);
            for (size_t i = 0; i < chunk_done.size(); ++i) chunk_rec[i].store(0);
        }
        prof_chunk = prof_on;
        const bool lm = lm_begin(prun);   // (one device only: the head's run is the run)
        struct LmEnd {   // (every way out: its host memory freed once the device is done with it)
            Impl* p;
            bool on;
            ~LmEnd() {
                if (on) p->lm_end();
            }
        } lm_end_{this, lm};
        tail_mode = tail_on;
        struct TailEnd {   // (every way out: no pass keeps the tail's source/filter rules)
            Impl* p;
            ~TailEnd() {
                p->tail_mode = false;
                p->chunk_prog_dev = -1;
            }
        } tail_end_{this};
        // one pass over the split: every device's worker, the progress from the device the prompt's chunks are
        // followed on (the tail pass reports nothing: only its last rows are the prompt's own reading)
        auto run_pass = [&](std::vector<Run>& rr, int prog, bool report) -> bool {
            chunk_prog_dev = report ? prog : -1;
            pass_devs.assign(devs.size(), 0);
            for (size_t di = 0; di < devs.size(); ++di) {
                bool any = false;
                const Run& gr = rr[(size_t) devs[di].grp];
                const bool stage1 = gr.stage == 1;
                if (!gr.T.empty())   // (a group without a run in this pass has no layer in it, whatever its stage says)
                    for (int il = 0; il < g.nl && !any; ++il) {
                        if (tail_mode && (stage1 ? il < tail_first() : il > g.cand_src)) continue;
                        for (int m = 0; m < L[(size_t) il].nm && !any; ++m) any = member(L[(size_t) il], m).dev == (int) di;
                    }
                pass_devs[di] = any ? 1 : 0;
            }
            std::vector<std::string> errs(devs.size());
            for (size_t di = 0; di < devs.size(); ++di)
                workers[di]->submit([this, di, &rr, &errs] {
                    std::string e;
                    if (!run_dev_chunks((int) di, rr, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
                });
            size_t reported = 0;
            const Run& rp = rr[(size_t) devs[(size_t) prog].grp];
            for (;;) {
                bool all = true;
                for (auto& w : workers) all = all && w->idle();
                if (report && progress && lm) {   // (layer-major: the share of the layers every chunk went through)
                    const int nd = lm_layers.load(std::memory_order_acquire);
                    if (nd > 0 && nd < g.nl) progress((int64_t) n * nd / g.nl);
                } else if (report && progress) {   // chunks through every layer (the progress device past their event)
                    while (reported < rp.T.size() && chunk_rec[reported].load(std::memory_order_acquire) &&
                           cudaEventQuery(chunk_done[reported]) == cudaSuccess)
                        ++reported;
                    if (reported > 0 && reported < rp.T.size())
                        progress(rp.p0[reported - 1] + rp.T[reported - 1] - rp.p0[0]);
                }
                if (all) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            for (const std::string& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
            return true;
        };
        live_base = pos;
        live_ntot = n;
        if (!run_pass(runs, prog0, true)) return false;
        if (tail_on) {   // all first-pass work landed: the tail's rows to the host, the source's rows to every reader
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
            if (!tail_sync_src(pos + n, err)) return false;
            if (pl_end_.on) {   // DS4_PASS_LEND: pass 2's borrow - the encoder layers' banks to the tail
                if (!pl_borrow(1, err)) return false;
                plan_streams();
            }
            if (!run_pass(truns, head_dev, false)) return false;
        }
        st.layer_ms += now_ms() - tl;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        if (prof_chunk) {   // DS4_PROF: the run's GPU time by stage, every device's summed
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
            cudaSetDevice(d.id);
            prof_collect();
            std::fprintf(stderr,
                         "ds4 prompt GPU time, %d tokens: attention %.0f ms, routing + shared expert %.0f ms, experts %.0f ms "
                         "(with the waits for streamed ones), between layers %.0f ms\n",
                         n, cprof_ms[1], cprof_ms[2], cprof_ms[3], cprof_ms[0]);
            for (double& v : cprof_ms) v = 0;
            prof_chunk = false;
        }
        if (ep_run && (o.verbose || std::getenv("DS4_EP_STATS") != nullptr)) {   // (its shares, layer by layer)
            const double ms = now_ms() - tl;
            for (size_t di = 0; di < chunks.size(); ++di) {
                const Chunk& c = *chunks[di];
                std::fprintf(stderr,
                             "ds4 ep, CUDA%d: x | ridx | rw and its acc through the host %.2f GB, %.1f GB of experts "
                             "streamed, %.1f ms\n",
                             devs[di].id, c.ep_bytes / 1e9, c.stream_bytes / 1e9, ms);
            }
        }
        if (o.verbose || std::getenv("DS4_FLOW_STATS") != nullptr) {
            const double ms = now_ms() - tl;
            for (size_t di = 0; di < chunks.size(); ++di) {
                const Chunk& c = *chunks[di];
                if (!c.flow || ep_run) continue;
                const double db = (double) (c.io->direct_bytes() - c.io_d0) / 1e9;
                const double bb = (double) (c.io->buffered_bytes() - c.io_b0) / 1e9;
                const double fb = db + bb, rb = (double) c.it_ram * (double) L[0].blob / 1e9;   // (a member: its parts)
                char lcs[160] = "";
                if (c.lm)
                    std::snprintf(lcs, sizeof lcs, ", %llu again from the layer cache (layer-major, %d chunks, %.1f GB "
                                  "of RAM)", (unsigned long long) c.it_lc,
                                  (int) runs[(size_t) devs[di].grp].T.size(), (double) c.lc_bytes / 1e9);
                std::fprintf(stderr,
                             "ds4 prompt stream, CUDA%d: %d tokens in %.0f ms; %llu experts streamed: %llu from the files "
                             "(%.1f GB, %.2f GB/s: %.1f GB from the disk unbuffered, %.1f GB through the page cache), %llu "
                             "from RAM (%.1f GB)%s; %llu likely ones queued early, %llu of them unpicked; the compute waited "
                             "%.0f ms for reads\n",
                             devs[di].id, n, ms, (unsigned long long) (c.it_file + c.it_ram + c.it_lc),
                             (unsigned long long) c.it_file, fb, fb / std::max(1e-3, ms / 1000.0), db, bb,
                             (unsigned long long) c.it_ram, rb, lcs, (unsigned long long) c.it_spec,
                             (unsigned long long) c.it_spec_unused, c.wait_ms);
            }
        }
        if (pm.on && !tail_on && frun.T.size() == 1) {   // (the picks' spread for pick_path: the CPU-lane experts this
            bool tp = false;                             // run picked - never the tail pass's own, it is not a prompt)
            for (const Layer& l : L) tp = tp || l.nm > 1;
            double picked = 0;
            for (const auto& cp : chunks)
                if (cp != nullptr && cp->flow) picked += (double) (cp->it_file + cp->it_ram) - (double) cp->it_spec_unused;
            if (!tp) pm_learn(n, picked);
        }
        if (logits_all) n_logits = (T_last - 1) % kWin + 1;   // (sample() takes the last row of d.logits)
        pos += n;
        st.tokens += n;
        if (pl_end_.on) {   // (the decode layout back: exit_prefill fills both sides of every pair again)
            pl_end_.on = false;
            if (!pl_restore(err)) return false;
        }
        if (!exit_prefill(err)) return false;   // (the decoder's experts back into the arena, the hottest first)
        cache_tick(n);
        int tap_n = T_last;   // the drafter's seed: the last rows of every tapping group's last chunk (the shortest one:
        for (int il : g.d_targets)   // a group's last chunk can be shorter than the head's - e.g. n = 8321)
            if (il >= 0 && il < (int) L.size())
                tap_n = std::min(tap_n, (tail_on ? truns : runs)[(size_t) devs[(size_t) L[(size_t) il].dev].grp].T.back());
        const int nr = std::min(tap_n, tap_cap);
        if (!seed_drafter(pos - nr, nr, 0, err)) return false;
        if (snap_slot >= 0) {   // checkpoint `snap_slot` from what the layers kept (the drafter's blocks: as they are -
            // the next feed seeds them again from its own rows, past this position)
            Check& ckp = ck[snap_slot & 1];
            ckp.pos = frun.snap_pos;
            ckp.win.resize(L.size());
            ckp.comp.resize(L.size() * 2);
            for (size_t i = 0; i < L.size(); ++i) {
                Layer& l = L[i];
                Dev& dv = devs[(size_t) l.dev];
                cudaSetDevice(dv.id);
                const bool kept = !l.mtp && l.snap_win != nullptr;
                ckp.win[i].resize((size_t) g.win * win_rb());
                CK(cudaMemcpyAsync(ckp.win[i].data(), kept ? l.snap_win : l.win, ckp.win[i].size(),
                                   cudaMemcpyDeviceToHost, dv.st));
                if (l.c_kv) {
                    ckp.comp[2 * i].resize((size_t) l.ratio * g.hd);
                    ckp.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                    CK(cudaMemcpyAsync(ckp.comp[2 * i].data(), kept && l.snap_kv ? l.snap_kv : l.c_kv,
                                       ckp.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                    CK(cudaMemcpyAsync(ckp.comp[2 * i + 1].data(), kept && l.snap_sc ? l.snap_sc : l.c_score,
                                       ckp.comp[2 * i + 1].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                }
            }
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
        }
        return true;
    }

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    void plan_window(int64_t p0, int T, WinPlan& wp) {
        wp.par = par;
        wp.async = par || (!sync_decode && T <= async_max_t);
        wp.vl = 0;
        for (int t = 0; t < T && !imgs.empty(); ++t)
            if (in_image(p0 + t)) wp.vl |= 1u << t;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.ratio > 0) {
                const int R = l.ratio;
                int n[kWin], k[kWin], clen_max = 0;
                for (int t = 0; t < T; ++t) {
                    n[t] = (int) ((p0 + t + 1) / R);
                    k[t] = std::min(g.ix_topk, n[t]);
                    clen_max = std::max(clen_max, n[t]);
                }
                if (l.ix_src) {
                    for (int t = 0; t < T; ++t) wp.cand[il][t] = cur_cand_n[t];
                    if (clen_max > 0 && l.cand_src)
                        for (int t = 0; t < T; ++t) cur_cand_n[t] = std::min(g.cand_topk, (n[t] + g.cand_blk - 1) / g.cand_blk);
                    for (int t = 0; t < T; ++t) cur_topk_n[t] = k[t];
                }
                for (int t = 0; t < T; ++t) wp.topk[il][t] = cur_topk_n[t];
            }
            if (wp.async) {
                wp.seq[il] = ++seq_issued;
                layer_of[wp.seq[il] % kRing].store(il, std::memory_order_release);
            }
        }
    }

    int head_nm() const { return (int) groups[(size_t) devs[(size_t) head_dev].grp].mem.size(); }
    // the head's columns on member mi of its group of nm: an EXL3 head's split128 of the vocabulary (mat's), a GGUF's
    // BF16 / Q8_0 head's even shares (mat_bf16's) - head_share's, its gather's and the markov steps' rows
    void head_cols(int nm, int mi, int& v0, int& vn) const {
        const Tensor* tw = pk.get("head.weight");
        if (tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0")) {
            vn = g.vocab / nm;
            v0 = mi * vn;
        } else
            split128(g.vocab, nm, mi, v0, vn);
    }
    // the head over `rows` (<= 16) normed rows xn on member mi of the head group (device d): its share of each row's
    // logits (columns [v0, v0 + vn), head_cols); the lead gathers the others' into d.logits (rows V apart) - two
    // slots by seq (Vmax apart, the widest share's), a member reusing one once the lead acknowledged it.  gather
    // false: each member keeps its share (the drafter's markov steps over the group, mk_split)
    bool head_share(Dev& d, int mi, const float* xn, int rows, std::string& err, bool gather = true) {
        const int V = g.vocab, D = g.dim, nm = head_nm();
        int v0 = 0, vn = 0, Vmax = 0, z = 0;
        head_cols(nm, mi, v0, vn);
        head_cols(nm, 0, z, Vmax);
        Group& G = groups[(size_t) d.grp];
        float* out = mi == 0 ? d.logits : d.logits_part;
        const size_t ld = mi == 0 ? (size_t) V : (size_t) vn;
        cudaSetDevice(d.id);
        // (an EXL3 head: as many activations at a time as its split-K partials fit the scratch - 8 for a shared head, 4
        // for a whole one, each pass reading the whole share; a GGUF's: 4 at a time, as before)
        const int per = d.head.fmt != 0 ? 4 : std::max(1, x3::mv_cols(d.head));
        for (int c = 0; c < rows; c += per)
            if (!mv(d, {job(d.head, xn + (size_t) c * D, out + (size_t) c * ld, std::min(per, rows - c), D, ld)}, err))
                return false;
        if (nm == 1 || !gather) return true;
        const unsigned seq = ++d.head_seq;
        const int slot = (int) (seq % 2);
        auto stage = [&](int m) { return G.lg_stage + ((size_t) slot * (nm - 1) + (m - 1)) * kWin * Vmax; };
        auto flag = [&](int m) { return G.lg_flag + ((size_t) slot * (nm - 1) + (m - 1)) * 32; };
        if (mi > 0) {
            if (seq > 2) wait_flags(G.lg_ack + (size_t) slot * 32, 1, 32, seq - 2, d.st);
            Xfer x;
            x.n = 1;
            x.src[0] = d.logits_part;
            x.dst[0] = stage(mi);
            x.bytes[0] = (size_t) rows * vn * 4;
            xfer_put(x, flag(mi), seq, d.xcnt, d.st);
            return true;
        }
        for (int m = 1; m < nm; ++m) {
            int m0 = 0, mn = 0;
            head_cols(nm, m, m0, mn);
            Xfer x;
            x.n = 1;
            x.src[0] = stage(m);
            x.dst[0] = d.logits_part;
            x.bytes[0] = (size_t) rows * mn * 4;
            xfer_get(x, flag(m), seq, d.st);
            CK(cudaMemcpy2DAsync(d.logits + (size_t) m0, (size_t) V * 4, d.logits_part, (size_t) mn * 4,
                                 (size_t) mn * 4, (size_t) rows, cudaMemcpyDeviceToDevice, d.st));
        }
        set_flag(G.lg_ack + (size_t) slot * 32, seq, d.st);
        return true;
    }

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const int D = g.dim, H = g.hc;
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
        return head_share(d, d.mi, d.x, n_logits, err);
    }

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps)
    bool tap(Dev& d, int il, int T, std::string& err) {
        const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
        if (g.n_mtp == 0 || it == g.d_targets.end()) return true;
        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
        const int D = g.dim;
        cudaSetDevice(d.id);
        hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
        CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, d.st));
        return true;
    }

    bool forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows = -1) {
        if (T < 1 || T > kWin) {
            err = "a window of " + std::to_string(T) + " tokens";
            return false;
        }
        if (pos + T > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < T; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        if (!ensure_decode(err)) return false;   // (a window after a prompt: the arena's experts back in it)
        const int64_t p0 = pos;
        for (int t = 0; t < T; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        // the Engram rows asked for now (the first Engram layer's first), collected right before their layer: what is
        // left of their reads by then is the window's Engram time
        const double te = now_ms();
        const int64_t eg_w0 = eg_wait_ns.load();
        for (int li = 0; li < pk.eg_layers; ++li) {
            EgBatch& b = *eg_win[(size_t) li];
            if (b.live) eg_finish(b);   // (a window that failed midway: its reads end before h_eg is reused)
            eg_issue(p0, T, li, nullptr, b);
        }
        st.engram_ms += now_ms() - te;
        const int D = g.dim;
        for (int t = 0; t < T; ++t)   // (h_emb is written again by the next window, after this one's last sync)
            std::memcpy(h_emb + (size_t) t * D, emb_row(p0 + t, tk[t]), (size_t) D * 2);
        plan_window(p0, T, wplan);
        prof_window = prof_on && T == 1;
        n_logits = std::min(logits_rows, T);
        const double tl = now_ms();
        if (par) {
            if (!forward_par(p0, T, err)) return false;
        } else if (!forward_serial(p0, T, err)) return false;
        st.layer_ms += now_ms() - tl;
        for (auto& b : eg_win)
            if (b->live) eg_finish(*b);
        st.engram_ms += (double) (eg_wait_ns.load() - eg_w0) / 1e6;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        prof_collect();
        pos += T;
        st.tokens += T;
        cache_tick(T);
        if (seed_rows != 0 && !seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err)) return false;
        return true;
    }

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    bool forward_serial(int64_t p0, int T, std::string& err) {
        const int D = g.dim, H = g.hc;
        int cur = L[0].dev;
        {
            Dev& d = devs[(size_t) cur];
            cudaSetDevice(d.id);
            CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
            embed(d.emb_rows, T, d.h, D, H, d.st);
            init_pre(d.pre, T, d.st);
        }
        std::vector<float> dump;
        if (dump_h != nullptr) dump.resize((size_t) T * g.nl * H * D);
        for (int il = 0; il < g.nl; ++il) {
            Layer& l = L[(size_t) il];
            if (l.dev != cur) {
                Dev& a = devs[(size_t) cur];
                Dev& b = devs[(size_t) l.dev];
                cudaSetDevice(a.id);
                const size_t nh = (size_t) T * H * D * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                             nc = (size_t) T * cand_stride() * 4;
                uint8_t* bb = a.bounce;
                CK(cudaMemcpyAsync(bb, a.h, nh, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh, a.pre, np, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np, a.topk, nk, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np + nk, a.cand, nc, cudaMemcpyDeviceToHost, a.st));
                CK(cudaEventRecord(a.ev_out, a.st));
                cudaSetDevice(b.id);
                CK(cudaStreamWaitEvent(b.st, a.ev_out, 0));
                CK(cudaMemcpyAsync(b.h, bb, nh, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.pre, bb + nh, np, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.topk, bb + nh + np, nk, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.cand, bb + nh + np + nk, nc, cudaMemcpyHostToDevice, b.st));
                cur = l.dev;
            }
            if (!tap(devs[(size_t) l.dev], il, T, err)) return false;
            if (!layer(l, 0, p0, T, wplan, err)) return false;
            if (dump_h != nullptr) {   // DS4_DUMP_H: every layer's residual stream, for tools/ds4/compare.py --layers
                Dev& dd = devs[(size_t) l.dev];
                std::vector<float> hv((size_t) T * H * D);
                cudaSetDevice(dd.id);
                CK(cudaMemcpyAsync(hv.data(), dd.h, hv.size() * 4, cudaMemcpyDeviceToHost, dd.st));
                CK(cudaStreamSynchronize(dd.st));
                for (int t = 0; t < T; ++t)
                    std::memcpy(dump.data() + ((size_t) t * g.nl + l.il) * H * D, hv.data() + (size_t) t * H * D,
                                (size_t) H * D * 4);
            }
        }
        if (dump_h != nullptr) std::fwrite(dump.data(), 4, dump.size(), dump_h);
        if (cur != head_dev) {
            err = "the head is not on the last layer's GPU";
            return false;
        }
        return head_rows_dev(head_dev, T, err);
    }

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err) {
        wplan.win = ++win_seq;
        std::vector<std::string> errs(devs.size());
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, p0, T, &errs] {
                std::string e;
                if (!run_dev((int) di, p0, T, wplan, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        for (auto& w : workers) w->wait();
        for (const std::string& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        return true;
    }

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {d.h, d.pre, d.topk, d.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc;
        bool started = false;
        std::vector<char> got(srcs.size(), 0);
        for (int il = 0; il < g.nl; ++il) {
            Layer& L0 = L[(size_t) il];
            int mi = -1;
            for (int m = 0; m < L0.nm; ++m)
                if (member(L0, m).dev == di) mi = m;
            if (mi < 0) continue;
            if (!started) {   // the window's input: the embedding rows, or the previous group's hand-off
                started = true;
                if (il == 0) {
                    CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    embed(d.emb_rows, T, d.h, D, H, s);
                    init_pre(d.pre, T, s);
                } else {
                    const Dev& a = devs[(size_t) L[(size_t) il - 1].dev];
                    xfer_get(hand_xfer(a.hand, d, T, false), a.hand_flag, wp.win, s);
                }
            }
            // the compressed KV / index keys another group's source layer wrote this window
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || got[(size_t) sidx]) continue;
                got[(size_t) sidx] = 1;
                Src& sr = srcs[(size_t) sidx];
                if (L[(size_t) sr.layer].grp == L0.grp) continue;
                const int64_t g_first = p0 / sr.ratio;
                const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                if (ng == 0) continue;
                Xfer x;
                if (sr.ckv[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage;
                    x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                    x.bytes[x.n++] = (size_t) ng * ckv_rb();
                }
                if (sr.ik[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage + x_rows(sr) * ckv_rb();
                    x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                    x.bytes[x.n++] = (size_t) ng * ik_rb();
                }
                if (x.n > 0) xfer_get(x, sr.x_flag, wp.win, s);
            }
            if (mi == 0 && !tap(d, il, T, err)) return false;
            if (!layer(L0, mi, p0, T, wp, err)) return false;
            live_line(di, 2, il, 0, 1, p0, T, "windows");
            if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, il, p0, T);
            const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
            if (last && mi == 0 && il + 1 < g.nl) xfer_put(hand_xfer(d.hand, d, T, true), d.hand_flag, wp.win, d.xcnt, s);
        }
        if (d.grp == devs[(size_t) head_dev].grp) return head_rows_dev(di, T, err);
        return true;
    }

    // a source's new cache rows (src on device `from`, bytes of them) to the same place of its other devices' copies,
    // through pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<uint8_t*>& copies, uint8_t* stage, cudaEvent_t ev, const uint8_t* src, size_t at,
                   size_t bytes, int from, std::string& err) {
        bool any = false;
        for (size_t od = 0; od < devs.size(); ++od) any = any || (copies[od] != nullptr && (int) od != from);
        if (!any) return true;
        Dev& d = devs[(size_t) from];
        CK(cudaMemcpyAsync(stage, src, bytes, cudaMemcpyDeviceToHost, d.st));
        CK(cudaEventRecord(ev, d.st));
        for (size_t od = 0; od < devs.size(); ++od) {
            if (copies[od] == nullptr || (int) od == from) continue;
            Dev& o2 = devs[od];
            cudaSetDevice(o2.id);
            CK(cudaStreamWaitEvent(o2.st, ev, 0));
            CK(cudaMemcpyAsync(copies[od] + at, stage, bytes, cudaMemcpyHostToDevice, o2.st));
        }
        cudaSetDevice(d.id);
        return true;
    }

    // layer L0 for the window, as group member mi (its share of the heads, of the output projection and of the FFNs;
    // the partial sums all-reduced across the group)
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ag = groups[(size_t) L0.grp].ar;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        prof_mark(d, 0);
        // Engram
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            eg_finish(*eg_win[(size_t) l.eg]);   // (asked for when the window began)
            uint8_t* src = h_eg + (size_t) l.eg * eg_cap * per;
            const int efmt = eg_fmt(l.eg);
            uint8_t* sc = nullptr;
            if (efmt == 1) {
                const size_t rb = (size_t) (ehd / 256) * 210;   // a column's Q6_K bytes, the slot's first ones
                for (int t = 0; t < T; ++t)   // (a token's slot is per bytes: its columns hd apart, not hd * cols)
                    CK(cudaMemcpy2DAsync(d.eg_rows + (size_t) t * cols * rb, rb, src + (size_t) t * per, (size_t) ehd,
                                         rb, (size_t) cols, cudaMemcpyHostToDevice, s));
            } else {
                for (int t = 0; t < T; ++t)   // [token][rows | scales] -> [all rows][all scales]
                    CK(cudaMemcpyAsync(d.eg_rows + (size_t) t * cols * ehd, src + (size_t) t * per, (size_t) cols * ehd,
                                       cudaMemcpyHostToDevice, s));
                sc = d.eg_rows + (size_t) T * cols * ehd;
                for (int t = 0; t < T; ++t)
                    CK(cudaMemcpyAsync(sc + (size_t) t * cols * (ehd / 32), src + (size_t) t * per + (size_t) cols * ehd,
                                       (size_t) cols * (ehd / 32), cudaMemcpyHostToDevice, s));
            }
            engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s, efmt);
            if (!mv(d, {job(l.eg_wkv, d.eg_x, d.eg_kv, T, (size_t) cols * ehd, (size_t) (H + 1) * D)}, err)) return false;
            if (std::getenv("DS4_DBG_HC") != nullptr && l.il <= 1)
                std::fprintf(stderr, "DS4_DBG_EG il %d ng %d heads %d hd %d cols %d fmt %d\n", l.il, pk.eg_ngram,
                             pk.eg_heads, ehd, cols, efmt);
            dbg_nan("eg_x", l.il, T, d.eg_x, (size_t) cols * ehd, 16, s);
            dbg_nan("eg_kv", l.il, T, d.eg_kv, (size_t) (H + 1) * D, 0, s);
            dbg_nan("h_pre_eg", l.il, T, d.h, (size_t) H * D, 0, s);
            engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);
            dbg_nan("h_post_eg", l.il, T, d.h, (size_t) H * D, 0, s);
        }
        // attention
        hc_mix(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, s);
        hc_split(d.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, d.a_pre, d.a_post, d.a_comb, s);
        hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);
        {
            x3::MvJob j[4];
            int nj = 0;
            j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
            j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
            if (l.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
            if (l.kv_src && l.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
            if (!mv_n(d, j, nj, err)) return false;
        }
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        finish(d.qr, T, g.q_lora, g.q_lora, fq, s);
        if (NHL > 0) {
            if (!mv(d, {job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)}, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) p0;
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(d.q, T * NHL, HD, HD, fr, s);
        }
        // the window rows: the ring's (positions p0 - win ..) then this window's, each query its own 128.  A DSpark
        // block's ring holds the trunk's main KV (positions <= p0 - 1, seeded); its block of drafts attends to all of
        // that ring and to the whole block, and leaves the ring as it was
        ring_to_ext(l.win, d.ext, l.mtp ? (int64_t) g.win : p0, g.win, HD, s);   // (a block's ring: its rows as they lie)
        Finish fk;
        fk.w = l.kv_norm;
        fk.eps = g.eps;
        fk.rd = g.rd;
        fk.pos = (int) p0;
        fk.freqs = freqs;
        fk.quant = 1;
        fk.dst = d.ext + (size_t) g.win * HD;
        fk.dst_stride = HD;
        finish(d.kvr, T, HD, HD, fk, s);
        if (!l.mtp) ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, spec_win ? l.ring_bak : nullptr);
        AttnRows ar;
        for (int t = 0; t < T; ++t) {
            if (l.mtp) {
                ar.win0[t] = 0;
                ar.nwin[t] = (int) std::min<int64_t>(g.win, p0);
                ar.nsel[t] = T;
                continue;
            }
            const int first = (int) std::max<int64_t>(g.win - p0, t + 1);   // ext row of position max(0, p - win + 1)
            ar.win0[t] = first;
            ar.nwin[t] = g.win + t - first + 1;
            ar.nsel[t] = 0;
        }
        const void* ckv = nullptr;
        int ckv_fmt = kKvFp4;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            // the groups this window completes: g_first .. g_first + ng - 1
            const int64_t g_first = p0 / R;              // the first group whose last position is >= p0
            const int64_t g_end = (p0 + T) / R;          // groups completed by p0 + T - 1
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(d.ckr, T, HD, HD, fc, s);
                    CK(cudaMemcpyAsync(d.latent, d.ckr, (size_t) T * HD * 4, cudaMemcpyDeviceToDevice, s));
                } else {
                    if (spec_win && l.c_snap != nullptr) {   // to replay the accepted rows if drafts are rejected
                        const size_t ns = (size_t) R * HD * 4, nr = (size_t) T * HD * 4;
                        CK(cudaMemcpyAsync(l.c_snap, l.c_kv, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_snap + (size_t) R * HD, l.c_score, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows, d.ckr, nr, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows + (size_t) kWin * HD, d.cgr, nr, cudaMemcpyDeviceToDevice, s));
                    }
                    compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, d.latent, s);
                }
            }
            Rows lim;
            int clen_max = 0;
            for (int t = 0; t < T; ++t) {
                lim.n[t] = (int) ((p0 + t + 1) / R);
                lim.k[t] = std::min(g.ix_topk, lim.n[t]);
                clen_max = std::max(clen_max, lim.n[t]);
            }
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    if (!mv(d, {job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                    if (!wp.par && !replicate(ks.ik, ks.h_ik, ks.ev_ik, fi.pdst, (size_t) g_first * ik_rb(),
                                              (size_t) ng * ik_rb(), l.dev, err))
                        return false;
                }
                if (clen_max > 0) {
                    if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.rd = g.rd;
                    fi.pos = (int) p0;
                    fi.vpp = g.ix_heads;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                    // (iw rows are 64 apart)
                    gemv_f16_rows(l.i_wproj, d.x, (size_t) D, g.ix_heads, D, T,
                                  1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, s);
                    const long long ss = (long long) o.max_context + 64;
                    bool restrict_ = false;
                    for (int t = 0; t < T; ++t)
                        restrict_ = restrict_ || (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (l.uses_cand && restrict_) {
                        Rows cl;   // the candidate positions of each query: its blocks x blk
                        int nmax = 0;
                        for (int t = 0; t < T; ++t) {
                            cl.n[t] = lim.n[t];
                            cl.k[t] = wp.cand[l.il][t];
                            nmax = std::max(nmax, wp.cand[l.il][t] * g.cand_blk);
                        }
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, nmax, d.cand,
                                    (int) cand_stride(), g.cand_blk, cl, d.scores, ss, s);
                        Rows tr;
                        for (int t = 0; t < T; ++t) {
                            tr.n[t] = wp.cand[l.il][t] * g.cand_blk;
                            tr.k[t] = lim.k[t];
                        }
                        topk(d.scores, ss, T, tr, d.cand, (int) cand_stride(), g.cand_blk, d.topk, (int) topk_stride(), s);
                    } else {
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1,
                                    lim, d.scores, ss, s);
                        if (l.cand_src) {
                            Rows cr;
                            for (int t = 0; t < T; ++t) {
                                cr.n[t] = lim.n[t];
                                cr.k[t] = g.cand_topk;
                            }
                            cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) bm_stride(), d.cand,
                                        (int) cand_stride(), s);
                        }
                        topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s);
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(d.latent, ng, HD, HD, fc, s);
                if (!wp.par && !replicate(sr.ckv, sr.h_ckv, sr.ev_ckv, fc.pdst, (size_t) g_first * ckv_rb(),
                                          (size_t) ng * ckv_rb(), l.dev, err))
                    return false;
                if (wp.par && mi == 0 && sr.remote) {   // the new rows for the other groups' readers
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.x_stage;
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.x_stage + x_rows(sr) * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.x_flag, wp.win, d.xcnt, s);
                }
            }
            for (int t = 0; t < T; ++t) ar.nsel[t] = lim.n[t] > 0 ? wp.topk[l.il][t] : 0;
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (l.mtp) {   // the block's rows (BF16), after the ring's in ext, as every query's selection
            ckv = d.ext + (size_t) g.win * HD;
            ckv_fmt = kKvBf16;
            iota_rows(d.topk, T, T, (int) topk_stride(), s);
        }
        if (NHL > 0) {
            sparse_attn(d.q, d.ext, ckv, d.topk, (int) topk_stride(), ar, l.sink, NHL, HD, T,
                        1.0f / std::sqrt((float) HD), d.o, s, ckv_fmt, 0, &d.attn_scr);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) p0;
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(d.o, T * NHL, HD, HD, fo, s);
            x3::MvJob j[16];
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                j[gi] = job(l.wo_a[gi], d.o + (size_t) gi * per, d.oa + (size_t) gi * g.o_lora, T, HQ, (size_t) OG * g.o_lora);
            if (!mv_n(d, j, OG, err)) return false;
            if (!mv(d, {job(l.wo_b, d.oa, d.ao, T, (size_t) OG * g.o_lora, D)}, err)) return false;
        } else CK(cudaMemsetAsync(d.ao, 0, (size_t) T * D * sizeof(*d.ao), s));   // (no heads: nothing to add)
        if (nm > 1) allreduce(d.ao, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
        hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        dbg_nan("attn_ao", l.il, T, d.ao, D, 0, s);
        dbg_nan("attn_h", l.il, T, d.h, (size_t) H * D, 0, s);
        dbg_nan("attn_mix", l.il, T, d.mix, 32, 24, s);
        dbg_nan("attn_pre", l.il, T, d.a_pre, 8, 4, s);
        dbg_nan("attn_post", l.il, T, d.a_post, 8, 4, s);
        dbg_nan("attn_comb", l.il, T, d.a_comb, 64, 16, s);
        dbg_nan("attn_scale", l.il, 1, l.hca_scale, 3, 3, s);
        dbg_nan("attn_base", l.il, 1, l.hca_base, 24, 24, s);
        if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, 1000 + l.il, p0, T);   // (DS4_DUMP_L: after the attention)
        prof_mark(d, 1);
        // the MoE
        hc_mix(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, s);
        hc_split(d.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, d.f_pre, d.f_post, d.f_comb, s);
        hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);
        if (wp.async) {
            // the routing goes to the service thread through the mailbox (a group's lead publishes it, the members
            // make the same plan) while the device runs the shared expert and the VRAM experts (from the plan the
            // router made), and waits for the answer only before the experts' sum
            gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
            const unsigned sq = wp.seq[l.il];
            const int slot = (int) (sq % kRing);
            const bool lead = mi == 0;
            int na[2] = {0, 0};   // lookahead: the next layers' guesses (their gate on this FFN input), published below
            for (int a = 1; lead && a <= std::min(2, ahead_mode); ++a) {
                const Layer* nx = ahead_layer(L0, a);
                if (nx == nullptr) break;
                const int m = a == 1 ? std::max(1, std::min(kAhead1, nx->k_exp + ahead_extra)) : std::min(kPicks, nx->k_exp);
                float* gl = d.glog_a + (size_t) (a - 1) * kWin * 1024;
                gemv_f32(nx->gate_w, d.x, nx->n_exp, D, T, gl, 1024, s);   // (normed with the next layer's FFN
                                                                         // weights instead: the same hit rate)
                route_ahead(gl, nx->gate_bias, nx->n_exp, m, T, reqs + slot, (a - 1) * kAhead1, s);
                na[a - 1] = m;
            }
            route_publish(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, l.vram_tab, d.x, D, d.ridx, d.rw,
                          d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl, l.mtp ? 0u : wp.vl, na[0],
                          na[1]);
            if (!mv(d, {job(l.s1, d.x, d.sg, T, D, FF), job(l.s3, d.x, d.su, T, D, FF)}, err)) return false;
            swiglu(d.sg, d.su, T * FF, g.lim, d.sh, s);
            if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, FF, D)}, err)) return false;
            prof_mark(d, 2);
            const x3::ExpertLayout& lay = nm > 1 ? l.hlay : l.lay;
            if (T == 1 && moe_token_path) {
                x3::MoeArgs ma;
                ma.plan_ptr = d.plan_ptr;
                ma.plan_w = d.rw;
                ma.cpu_part = lead ? resps[slot].part[0] : nullptr;
                ma.cpu_flag = lead ? &resps[slot].cpu[0] : nullptr;
                ma.lay = lay;
                ma.k = l.k_exp;
                ma.n_embd = D;
                ma.n_ff = FF;
                ma.limit = g.lim;
                const bool mx = lay.fmt == gg::kFmtMXFP4;
                if (mx) gg::moe_gate_up(ma, d.x, d.xg, s);
                else x3::moe_gate_up(ma, d.x, d.xg, s);
                prof_mark(d, 3);
                if (lead) wait_seq(&resps[slot].seq, sq, s);
                prof_mark(d, 4);
                if (mx) gg::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
                else x3::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
            } else {   // each distinct expert read once for the window's tokens
                gf::MoeWin mw;
                mw.plan_ptr = d.plan_ptr;
                mw.T = T;
                mw.k = l.k_exp;
                mw.grp_ptr = d.grp_ptr;
                mw.grp_start = d.grp_start;
                mw.grp_ent = d.grp_ent;
                mw.n_groups = d.n_groups;
                gf::moe_win_group(mw, s);
                x3::WinArgs wa;
                wa.grp_ptr = d.grp_ptr;
                wa.grp_start = d.grp_start;
                wa.grp_ent = d.grp_ent;
                wa.n_groups = d.n_groups;
                wa.plan_ptr = d.plan_ptr;
                wa.plan_w = d.rw;
                wa.cpu_part = lead ? &resps[slot].part[0][0] : nullptr;
                wa.cpu_stride = kMaxDim;
                wa.cpu_flag = lead ? resps[slot].cpu : nullptr;
                wa.wait_ptr = lead ? &resps[slot].seq : nullptr;
                wa.wait_val = sq;
                wa.x = d.x;
                wa.x_stride = (size_t) D;
                wa.h = d.xg;
                wa.part = d.ed;
                wa.sh_out = d.sh_out;
                wa.out = d.moe_out;
                wa.out_stride = (size_t) D;
                wa.lay = lay;
                wa.T = T;
                wa.k = l.k_exp;
                wa.n_embd = D;
                wa.n_ff = FF;
                wa.limit = g.lim;
                if (const char* dbg = std::getenv("DS4_DBG_PLAN");
                    dbg != nullptr && lead && (gg::expert_type(lay, 0) == 17 || gg::expert_type(lay, 0) == 18)) {   // (the window's plan)
                    CK(cudaStreamSynchronize(s));
                    const size_t nb = (size_t) T * l.k_exp, nb8 = (size_t) T * kPicks, np = (size_t) kWin * kPicks;
                    std::vector<unsigned long long> hp(np), hv((size_t) l.n_exp);
                    std::vector<int> hi(np);
                    std::vector<float> hw(np), hlog((size_t) T * 1024), hxin((size_t) T * D);
                    int ng = 0;
                    CK(cudaMemcpy(hp.data(), d.plan_ptr, nb8 * 8, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hv.data(), l.vram_tab, hv.size() * 8, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hi.data(), d.ridx, nb8 * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hw.data(), d.rw, nb8 * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hlog.data(), d.glog, hlog.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hxin.data(), d.x, hxin.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&ng, d.n_groups, 4, cudaMemcpyDeviceToHost));
                    const unsigned long long vb = (unsigned long long) l.vram_base,
                                               ve = vb + (unsigned long long) l.n_perm * unit(l),
                                               lb = (unsigned long long) l.lend_base,
                                               le = lb + (unsigned long long) l.n_lend * unit(l);
                    std::fprintf(stderr, "DS4_DBG_PLAN il %d T %d k %d types %d/%d/%d n_vram %d n_slots %d vbase %p ng %d\n",
                                 l.il, T, l.k_exp, gg::expert_type(lay, 0), gg::expert_type(lay, 1),
                                 gg::expert_type(lay, 2), l.n_vram, l.n_slots, (void*) l.vram_base, ng);
                    int nbad = 0;
                    for (size_t e = 0; e < hv.size(); ++e)
                        if (hv[e] != 0 && !((hv[e] >= vb && hv[e] < ve) || (hv[e] >= lb && hv[e] < le))) {
                            if (nbad < 8) std::fprintf(stderr, "  bad vram_tab[%zu] %llx\n", e, hv[e]);
                            ++nbad;
                        }
                    std::fprintf(stderr, "  vram_tab: %d bad of %zu\n", nbad, hv.size());
                    for (int t = 0; t < T; ++t) {
                        int nl = 0, nx = 0;
                        for (int i = 0; i < 1024; ++i) nl += hlog[(size_t) t * 1024 + i] != hlog[(size_t) t * 1024 + i];
                        for (int i = 0; i < D; ++i) nx += hxin[(size_t) t * D + i] != hxin[(size_t) t * D + i];
                        std::fprintf(stderr, "  t %d glog nan %d [%g %g %g %g] x nan %d [%g %g %g %g]\n", t, nl,
                                     hlog[(size_t) t * 1024], hlog[(size_t) t * 1024 + 1],
                                     hlog[(size_t) t * 1024 + 2], hlog[(size_t) t * 1024 + 3], nx,
                                     hxin[(size_t) t * D], hxin[(size_t) t * D + 1], hxin[(size_t) t * D + 2],
                                     hxin[(size_t) t * D + 3]);
                    }
                    for (size_t i = 0; i < nb8; ++i)
                        std::fprintf(stderr, "  plan[%zu] %llx pick %d w %g\n", i, hp[i], hi[i], hw[i]);
                    std::vector<float> hmo((size_t) T * D), hsh((size_t) T * D), hh((size_t) T * H * D);
                    CK(cudaMemcpy(hmo.data(), d.moe_out, hmo.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hsh.data(), d.sh_out, hsh.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hh.data(), d.h, hh.size() * 4, cudaMemcpyDeviceToHost));
                    for (int t = 0; t < T; ++t) {
                        int nm_ = 0, ns = 0, nh = 0;
                        for (int i = 0; i < D; ++i) {
                            nm_ += hmo[(size_t) t * D + i] != hmo[(size_t) t * D + i];
                            ns += hsh[(size_t) t * D + i] != hsh[(size_t) t * D + i];
                        }
                        for (int i = 0; i < H * D; ++i) nh += hh[(size_t) t * H * D + i] != hh[(size_t) t * H * D + i];
                        int nc = 0;
                        for (int i = 0; i < D; ++i)
                            nc += resps[(size_t) slot].part[t][i] != resps[(size_t) slot].part[t][i];
                        std::fprintf(stderr, "  prev t %d moe_out nan %d sh_out nan %d h nan %d cpu %d cpu nan %d [%g %g]\n", t,
                                     nm_, ns, nh, resps[(size_t) slot].cpu[t], nc, hmo[(size_t) t * D],
                                     hmo[(size_t) t * D + 1]);
                    }
                }
                if (!(lay.fmt == gg::kFmtMXFP4 ? gg::win_experts(wa, s) : x3::win_experts(wa, s))) {
                    err = "exl3 win_experts failed";
                    return false;
                }
            }
            if (lead) st.experts += (int64_t) T * l.k_exp;
            if (nm > 1) allreduce(d.moe_out, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
            hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
            std::swap(d.h, d.h2);
            dbg_nan("moe_out", l.il, T, d.moe_out, D, 0, s);
            dbg_nan("moe_h", l.il, T, d.h, (size_t) H * D, 0, s);
            dbg_nan("f_mix", l.il, T, d.mix, 32, 24, s);
            dbg_nan("f_pre", l.il, T, d.f_pre, 8, 4, s);
            dbg_nan("f_post", l.il, T, d.f_post, 8, 4, s);
            dbg_nan("f_comb", l.il, T, d.f_comb, 64, 16, s);
            dbg_nan("ffn_scale", l.il, 1, l.hcf_scale, 3, 3, s);
            dbg_nan("ffn_base", l.il, 1, l.hcf_base, 24, 24, s);
            CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            prof_mark(d, 5);
            return true;
        }
        if (nm > 1) {
            err = "a tensor-parallel layer on the host-routed path";
            return false;
        }
        // (glog rows are 1024 apart: route reads them so)
        gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
        VlBits vb = {};
        vb.w[0] = l.mtp ? 0u : wp.vl;
        route(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, d.ridx, d.rw, s, l.gate_bias_vl,
              vb.w[0] != 0 ? &vb : nullptr);
        CK(cudaMemcpyAsync(h_idx, d.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_w, d.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_x, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(d.ev, s));
        // the shared expert while the host looks at the picks
        if (!mv(d, {job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)}, err)) return false;
        swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);
        if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, g.ff, D)}, err)) return false;
        const double tw = now_ms();
        CK(cudaEventSynchronize(d.ev));
        st.gpu_wait_ms += now_ms() - tw;
        // the picks by expert: the VRAM ones' rows (gathered inputs, one GEMV per matrix for all their tokens), the
        // CPU lane's entries
        Combine cm;
        Gather ga;
        uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        int vex[kWin * kPicks], vrow0[kWin * kPicks], vn[kWin * kPicks];
        int nvex = 0, nrows = 0;
        std::vector<Ent> ents;
        std::vector<std::pair<int, float>> cpu_rows;   // slot -> (token, weight)
        std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);   // expert -> (token, pick)
        for (int t = 0; t < T; ++t) {
            cm.n[t] = 0;
            cm.cpu[t] = 0;
            for (int i = 0; i < l.k_exp; ++i) {
                const int e = h_idx[t * kPicks + i];
                if (e < 0 || e >= l.n_exp) {
                    err = "router picked expert " + std::to_string(e);
                    return false;
                }
                ++pr[e];
                l.heat[(size_t) e] += 1.0f;
                if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                per[(size_t) e].push_back({t, i});
            }
        }
        int nslots = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            const auto& pe = per[(size_t) e];
            if (pe.empty()) continue;
            if (l.slot[(size_t) e] >= 0) {
                vex[nvex] = e;
                vrow0[nvex] = nrows;
                vn[nvex] = (int) pe.size();
                for (const auto& tp : pe) {
                    ga.src[nrows] = tp.first;
                    const int t = tp.first;
                    cm.row[t][cm.n[t]] = nrows;
                    cm.w[t][cm.n[t]] = h_w[t * kPicks + tp.second];
                    ++cm.n[t];
                    ++nrows;
                }
                ++nvex;
            } else {
                for (size_t a = 0; a < pe.size(); a += kc::kExl3CpuMaxN) {
                    Ent en;
                    en.e = e;
                    en.slot0 = nslots;
                    en.n = (int) std::min<size_t>(kc::kExl3CpuMaxN, pe.size() - a);
                    for (int i = 0; i < en.n; ++i) {
                        en.tok[i] = pe[a + (size_t) i].first;
                        cpu_rows.push_back({pe[a + (size_t) i].first, h_w[pe[a + (size_t) i].first * kPicks + pe[a + (size_t) i].second]});
                        cm.cpu[en.tok[i]] = 1;
                    }
                    nslots += en.n;
                    ents.push_back(en);
                }
            }
        }
        st.experts += (int64_t) T * l.k_exp;
        st.cpu_experts += nslots;
        if (nrows > 0) {
            gather_rows(d.x, ga, nrows, D, d.xe, s);
            x3::MvJob j[2 * kWin * kPicks];
            int nj = 0;
            for (int i = 0; i < nvex; ++i) {
                const auto& vm = l.vm[(size_t) vex[i]];
                j[nj++] = job(vm[0], d.xe + (size_t) vrow0[i] * D, d.xg + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
                j[nj++] = job(vm[1], d.xe + (size_t) vrow0[i] * D, d.xu + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
            }
            if (!mv_n(d, j, nj, err)) return false;
            swiglu(d.xg, d.xu, nrows * g.ff, g.lim, d.xhh, s);
            nj = 0;
            for (int i = 0; i < nvex; ++i)
                j[nj++] = job(l.vm[(size_t) vex[i]][2], d.xhh + (size_t) vrow0[i] * g.ff, d.ed + (size_t) vrow0[i] * D,
                              vn[i], g.ff, D);
            if (!mv_n(d, j, nj, err)) return false;
        }
        if (!ents.empty()) {
            const double tc = now_ms();
            cpu_experts(l, h_x, ents);
            std::fill(h_part, h_part + (size_t) T * D, 0.0f);
            for (int sl = 0; sl < nslots; ++sl) {
                float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
                const float wv = cpu_rows[(size_t) sl].second;
                const float* y = c_yd.data() + (size_t) sl * D;
                for (int e = 0; e < D; ++e) o[e] += wv * y[e];
            }
            st.cpu_ms += now_ms() - tc;
            CK(cudaMemcpyAsync(d.cpu_part, h_part, (size_t) T * D * 4, cudaMemcpyHostToDevice, s));
        }
        combine(d.ed, ents.empty() ? nullptr : d.cpu_part, d.sh_out, cm, T, D, d.moe_out, s);
        hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        // (h_part / h_x are written again only after the next layer's route has synchronized, which follows these)
        return true;
    }

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::kernels::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        strata::kernels::sample_tokens(d.logits + (size_t) std::max(0, n_logits - 1) * g.vocab, 1, g.vocab, nullptr, 0,
                                       sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    bool logits(std::vector<float>& out) {
        if (logits_all) {
            out = chunk_logits;
            return true;
        }
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        out.resize((size_t) std::max(1, n_logits) * g.vocab);
        return cudaMemcpy(out.data(), d.logits, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void save_profile() const {
        if (o.profile.empty()) return;
        const std::string tmp = o.profile + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) return;
        const bool ok = std::fwrite(prof.data(), 4, prof.size(), f) == prof.size();
        std::fclose(f);
        // (the old profile replaced: std::rename does not replace an existing file on Windows, so there the profile
        // never moved on and only .tmp files were left)
        if (!ok || !os::replace_file(tmp, o.profile)) std::remove(tmp.c_str());
    }

    std::vector<std::string> info_lines() const {
        std::vector<std::string> v;
        int nh = 0;
        for (const Layer& l : L) nh += l.n_exp - l.n_vram;
        char b[512];
        // kv=fp4: DeepSeek's own cache layout (the compressed KV and the index keys FP4, the window FP8)
        std::snprintf(b, sizeof b, "INFO context=%lld kv=fp4 kv_resident=%lld expert_slots=%d engine_kind=ds4 "
                                   "cpu_experts=%d ram_experts=%d engine=ds4-0.1",
                      (long long) o.max_context, (long long) o.max_context, g.nl * g.n_exp - nh, nh, ram_held);
        v.push_back(b);
        if (rp.on) {   // (the row-split prefill: the group every card holds, how many groups the trunk is, its layers)
            std::snprintf(b, sizeof b, "INFO row_split=1 row_split_k=%d row_split_groups=%d row_split_layers=%d", rp.k,
                          rp.ng, g.nl);
            v.push_back(b);
        }
        // the split's GPUs, one line a card (the Monitor's GPUs and Loan tabs): the layers on it, the experts it holds
        // for good, the drafter's and the trunk's slots the prompt arena lends it between prompts, the arena itself and
        // the chunk its prompt path takes.  A line each, whatever the count - the server reads them as they come.
        int perm_all = 0, lend_all = 0, mtp_all = 0;
        int chunk_all = chunk_steps.empty() ? 0 : *std::max_element(chunk_steps.begin(), chunk_steps.end());
        std::string steps;
        for (size_t gi = 0; gi < chunk_steps.size(); ++gi)
            steps += (gi > 0 ? "," : "") + std::to_string(chunk_steps[gi]);
        for (size_t di = 0; di < devs.size(); ++di) {
            const Dev& d = devs[di];
            int a = -1, b2 = -1, perm = 0, lend = 0, mtp = 0, ev = 0;
            size_t perm_b = 0, lend_b = 0, mtp_b = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    if (x.dev != (int) di) continue;
                    if (a < 0) a = l.il;
                    b2 = l.il;
                    ev += l.n_exp;
                    if (l.mtp) {
                        mtp += l.n_lend;
                        mtp_b += (size_t) l.n_lend * unit(l);
                    } else {
                        perm += x.n_perm;
                        lend += x.n_lend;
                        perm_b += (size_t) x.n_perm * unit(l);
                        lend_b += (size_t) x.n_lend * unit(l);
                    }
                }
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            const int ch = d.grp < (int) chunk_steps.size() ? chunk_steps[(size_t) d.grp] : 0;
            std::snprintf(b, sizeof b,
                          "LEND dev=%d pci=%s total=%zu free=%zu weights=%zu caches=%zu arena=%zu perm=%d lend=%d mtp=%d "
                          "perm_mib=%zu lend_mib=%zu mtp_mib=%zu experts=%d chunk=%d layers=%d-%d",
                          d.id, d.pci.empty() ? "-" : d.pci.c_str(), tot >> 20, fr >> 20, d.weights >> 20,
                          d.caches >> 20, d.arena_bytes >> 20, perm, lend, mtp, perm_b >> 20, lend_b >> 20, mtp_b >> 20,
                          ev, ch, a, b2);
            v.push_back(b);
            perm_all += perm;
            lend_all += lend;
            mtp_all += mtp;
        }
        // the split's top line (the Monitor's tabs sum these, never a fixed card count): the GPUs the engine found, the
        // CPU lane's threads, and the chunk steps in the split's order
        std::snprintf(b, sizeof b,
                      "INFO gpus=%zu expert_lend=%d cpu_threads=%d prefill_chunk=%d prefill_steps=%s prefill_perm=%d "
                      "prefill_lend=%d prefill_mtp=%d",
                      devs.size(), o.expert_lend ? 1 : 0, cpu_threads(), chunk_all, steps.c_str(), perm_all, lend_all,
                      mtp_all);
        v.push_back(b);
        if (commit_kept > 0) {   // (Windows, an integrated GPU: the commit kept experts off the GPU - the server says so)
            std::snprintf(b, sizeof b,
                          "NOTE experts: %.0f GB stay off the GPU - Windows' page file is too small to hold them there; "
                          "a page file of %.0f GB keeps them on the GPU (System > About > Advanced system settings > "
                          "Performance > Advanced > Virtual memory), then restart Maya", commit_kept / 1e9, pagefile_hint);
            v.push_back(b);
            std::snprintf(b, sizeof b, "INFO gpu_commit_capped_gb=%.0f pagefile_hint_gb=%.0f", std::ceil(commit_kept / 1e9),
                          pagefile_hint);
            v.push_back(b);
        }
        if (pk.eg_layers > 0) {   // the n-gram memory's (Engram's) table rows a token reads, for the Monitor
            std::snprintf(b, sizeof b,
                          "INFO engram_layers=%d engram_lookups=%d engram_ngram=%d engram_heads=%d engram_cache_rows=%lld",
                          pk.eg_layers, pk.eg_layers * (pk.eg_ngram - 1) * pk.eg_heads, pk.eg_ngram, pk.eg_heads,
                          (long long) eg_cache.capacity());
            v.push_back(b);
        }
        if (g.n_mtp > 0) {   // the DSpark drafter: its block (a seed and the noise after it), its blocks, its targets
            std::snprintf(b, sizeof b, "INFO dspark_block=%d dspark_blocks=%d dspark_targets=%zu", g.d_block, g.n_mtp,
                          g.d_targets.size());
            v.push_back(b);
        }
        if (img_tok >= 0) {   // the server's image resize (DeepSeek's image_processor.py)
            std::snprintf(b, sizeof b,
                          "INFO vision=ds4 vision_patch=%d vision_merge=%d vision_min_pixels=%lld vision_max_tokens=%d "
                          "vision_max_wh_ratio=%lld image_token=%d vision_gpu=%d vision_mib=%lld",
                          vg.patch, vg.merge, (long long) pk.num("vision.min_pixels", 0), vg.max_tokens,
                          (long long) pk.num("vision.max_wh_ratio", 0), img_tok, vt.device(),
                          (long long) (vt.device_bytes() >> 20));
            v.push_back(b);
        } else if (!vision_err.empty()) {   // (the server offers no pictures and says why)
            std::string why = vision_err;
            for (char& ch : why)
                if (ch == '\n' || ch == '\r') ch = ' ';
            v.push_back("NOTE images: OFF - the model's vision tower did not load (" + why + ")");
            v.push_back("INFO vision=off");
        }
        return v;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
bool Model::load(const Options& o, std::string& err) { return p_->load(o, err); }
bool Model::step(int token, bool logits, std::string& err) { return p_->step(token, logits, err); }
bool Model::feed(const int* tokens, int n, int logits_rows, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    // the row-split prefill (--row-split-prefill): a prompt in the mode's window takes it, whatever pick_path predicts
    if (p.rp.on && n >= p.rp_min_rows() && logits_rows <= kWin) {
        std::string why;
        if (p.rp_ready(n, logits_rows, why)) {
            const double t0 = now_ms();
            const bool ok = p.forward_rowsplit(tokens, n, logits_rows, err);
            if (ok) p.path_done(now_ms() - t0);
            return ok;
        }
    }
    if (p.chunk > 0 && n >= p.chunk_min) {   // a long run - or one the flow could take, if it is the faster path here
        const double t0 = now_ms();   // (every row's logits (tests): the chunked path, which keeps them all)
        const bool ok = logits_rows > kWin || p.pick_path(n) ? p.forward_chunks(tokens, n, logits_rows, err)
                                                             : p.feed_windows(tokens, n, logits_rows, err);
        if (ok) p.path_done(now_ms() - t0);
        return ok;
    }
    p.st.path = 0;   // (windows of 16: the Monitor's Prefill tab)
    return p.feed_windows(tokens, n, logits_rows, err);
}
bool Model::feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    const double t0 = now_ms();
    if (n >= 2 && p.rp.on && n - 1 >= p.rp_min_rows() && logits_rows <= kWin) {   // (the row split, then the mark)
        std::string why;
        if (p.rp_ready(n - 1, 0, why)) {
            if (!p.forward_rowsplit(tokens, n - 1, 0, err)) return false;
            p.checkpoint(slot);
            const bool ok = p.feed_windows(tokens + n - 1, 1, logits_rows, err);
            if (ok) {
                p.st.path = 2;   // (the Monitor's Prefill tab: the prompt rode the row split, the last token a window)
                p.path_done(now_ms() - t0);
            }
            return ok;
        }
    }
    if (n >= 2 && p.chunk > 0 && n >= p.chunk_min && p.pick_path(n)) {   // (the last token rides along; its layers
        const bool ok = p.forward_chunks(tokens, n, logits_rows, err, slot);   // keep the state before it)
        if (ok) p.path_done(now_ms() - t0);
        return ok;
    }
    if (n > 1 && !p.feed_windows(tokens, n - 1, 0, err)) return false;
    p.st.path = 0;   // (windows of 16: the Monitor's Prefill tab)
    p.checkpoint(slot);
    const bool ok = n < 1 || p.feed_windows(tokens + n - 1, 1, logits_rows, err);
    if (ok) p.path_done(now_ms() - t0);   // (windows, when pick_path chose them)
    return ok;
}
int Model::feed_unit() const { return p_->chunk > 0 ? std::max(p_->chunk, (int) p_->o.max_context) : kWin; }
void Model::set_progress(std::function<void(int64_t)> f) { p_->progress = std::move(f); }
void Model::set_live(bool on) { p_->live_on.store(on); }
int Model::sample(float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample(t, tp, tk, mp, seed, counter);
}
bool Model::logits(std::vector<float>& out) { return p_->logits(out); }
int Model::image_token() const { return p_->img_tok; }
int Model::dim() const { return p_->g.dim; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err) {
    return p_->encode_image(rgb, width, height, span, err);
}
void Model::set_images(std::vector<Image> images) {
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) { return a.start < b.start; });
    p_->imgs = std::move(images);
}
int Model::draft_block() const { return p_->g.n_mtp > 0 ? p_->g.d_block : 0; }
bool Model::draft(int y, std::vector<int>& d, std::vector<float>& c, std::string& err, float t, float tp, int tk,
                  float mp, uint64_t seed, uint64_t counter) {
    return p_->draft(y, d, c, err, t, tp, tk, mp, seed, counter);
}
bool Model::verify(int y, const std::vector<int>& d, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter,
                   int& accepted, int& next, std::string& err) {
    return p_->verify(y, d, t, tp, tk, mp, seed, counter, accepted, next, err);
}
int64_t Model::pos() const { return p_->pos; }
int64_t Model::max_context() const { return p_->o.max_context; }
int Model::vocab() const { return p_->g.vocab; }
const std::vector<int32_t>& Model::tokens() const { return p_->toks; }
void Model::reset() { p_->reset(); }
void Model::checkpoint(int slot) { p_->checkpoint(slot); }
bool Model::restore(int64_t limit) { return p_->restore(limit); }
int64_t Model::restorable(int64_t limit) const { return p_->restorable(limit); }
void Model::save_profile() const { p_->save_profile(); }
const Stats& Model::stats() const { return p_->st; }
std::string Model::describe() const { return p_->describe(); }
std::vector<std::string> Model::info_lines() const { return p_->info_lines(); }

}  // namespace strata::ds4
