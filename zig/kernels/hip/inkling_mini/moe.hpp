#pragma once
#include "engine.hpp"
#include <string>

namespace inkling_moe {
constexpr int H = 2304, F = 1152, N = 128, K = 8;

__device__ inline float bf(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffU) >= 0x7f800000U) return x;
    return __uint_as_float((u + 0x7fffU + ((u >> 16) & 1)) & 0xffff0000U);
}

// Checkpoint float8 is E4M3FN, not the AMD FNUZ variant.
__device__ inline float e4m3(unsigned byte) {
    unsigned mag = byte & 127, exp = mag >> 3, frac = mag & 7;
    float v = exp ? __uint_as_float(((exp + 120) << 23) | (frac << 20)) : frac / 512.f;
    if (mag == 127) return __uint_as_float(0x7fc00000U);
    return byte & 128 ? -v : v;
}

__device__ inline float e2m1(unsigned code) {
    constexpr float values[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return code & 8 ? -values[code & 7] : values[code & 7];
}

// Four independent 32-lane row reductions per 128-thread block.
__device__ inline float sum32(float v) {
    for (int offset = 16; offset; offset >>= 1) v += __shfl_down(v, offset, 32);
    return v;
}

// Raw FP32 router dots. The router's linear is not a BF16-output mv, and
// select() applies the sigmoid, so batched prefill can feed it GEMM logits.
static __global__ void router(const float* w, const float* x, float* logits) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    float v = 0.f;
    for (int c = lane; c < H; c += 32) v = fmaf(w[row * H + c], x[c], v);
    v = sum32(v);
    if (!lane) logits[row] = v;
}

// One block of N threads per token: the lanes fetch in parallel, lane 0 ranks.
// A single lane chasing 256 dependent loads costs tens of microseconds.
static __global__ void select(const float* logits, const float* bias, int* ids, float* mix) {
    __shared__ float scores[N], offsets[N];
    int token = blockIdx.x;
    scores[threadIdx.x] = 1.f / (1.f + expf(-logits[token * N + threadIdx.x]));
    offsets[threadIdx.x] = bias[threadIdx.x];
    __syncthreads();
    if (threadIdx.x) return;
    int best[K];
    float ranks[K];
    for (int j = 0; j < K; ++j) { ranks[j] = -INFINITY; best[j] = 0; }
    for (int i = 0; i < N; ++i) {
        float rank = scores[i] + offsets[i];
        int j = K - 1;
        if (rank <= ranks[j]) continue;
        while (j && rank > ranks[j-1]) {
            best[j] = best[j-1]; ranks[j] = ranks[j-1]; --j;
        }
        best[j] = i; ranks[j] = rank;
    }
    float total = 0.f;
    for (int j = 0; j < K; ++j) total += scores[best[j]];
    // HF eager visits hit experts in ascending expert ID for the FP32 index_add_.
    for (int j = 0; j < K; ++j) {
        int pos = 0;
        for (int i = 0; i < K; ++i) pos += best[i] < best[j];
        ids[token * K + pos] = best[j];
        mix[token * K + pos] = (scores[best[j]] / (total + 1e-20f)) * 8.f;
    }
}

struct Packed {
    const unsigned char *w, *scale;
    const float *amax, *denom;
};
// Weights carried by one 32-bit load: eight NVFP4 nibbles or four MXFP8 bytes.
template<bool NV> constexpr int group = NV ? 8 : 4;

// The pinned loader materializes all converter inputs in the target BF16 dtype
// before Inkling2LoopDequantizeExperts does its FP32 arithmetic.
template<bool NV>
__device__ inline float outer_scale(const Packed& p, int expert) {
    return NV ? bf(p.amax[expert]) / bf(p.denom[0]) : 1.f;
}
// BF16 weights first..first+group-1 of the flattened [expert][row][col] tensor.
// `first` is a multiple of group, so the word load is aligned.
template<bool NV>
__device__ inline void unpack(const Packed& p, int first, float factor, float* w) {
    unsigned word = *reinterpret_cast<const unsigned*>(p.w + (NV ? first / 2 : first));
    float scale = NV ? e4m3(p.scale[first / 16]) : exp2f(float(p.scale[first / 32]) - 127.f);
    #pragma unroll
    for (int m = 0; m < group<NV>; ++m) {
        float v;
        if (NV) { v = e2m1((word >> (4 * m)) & 15) * scale; v *= factor; }
        else v = e4m3((word >> (8 * m)) & 255) * scale;
        w[m] = bf(v);
    }
}

// y[slot] = one expert row-block times one input row; H output rows for both
// projections (2*F == H). ids selects an expert per slot, or is null when every
// slot uses fixed_expert. stride is 0 when all slots share the same input.
//
// A row has 32 FMA chains, one per column residue mod 32, reduced by sum32's
// tree. Each thread owns `group` adjacent chains so a single load feeds all of
// them; a lane per column spent its time on byte loads (about 2x slower here)
// and wider threads ran out of registers. The arithmetic is unchanged.
template<bool NV, bool Down>
static __global__ void project(Packed p, const int* ids, int fixed_expert, const float* input,
                               int stride, float* y) {
    typedef float quad __attribute__((ext_vector_type(4)));
    constexpr int cols = Down ? F : H, W = group<NV>, per_row = 32 / W;
    int slot = blockIdx.y, expert = ids ? ids[slot] : fixed_expert;
    int row = blockIdx.x * (128 / per_row) + threadIdx.x / per_row, part = threadIdx.x % per_row;
    int base = (expert * H + row) * cols + part * W;
    const float* x = input + size_t(slot) * stride + part * W;
    float factor = outer_scale<NV>(p, expert), acc[W], w[W];
    for (int m = 0; m < W; ++m) acc[m] = 0.f;
    for (int c = 0; c < cols; c += 32) {
        unpack<NV>(p, base + c, factor, w);
        #pragma unroll
        for (int g = 0; g < W; g += 4) {
            quad xs = *reinterpret_cast<const quad*>(x + c + g);
            #pragma unroll
            for (int m = 0; m < 4; ++m) acc[g+m] = fmaf(w[g+m], xs[m], acc[g+m]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) { // sum32 over chains part*W+m
        if (offset >= W) { for (int m = 0; m < W; ++m) acc[m] += __shfl_down(acc[m], offset / W, per_row); }
        else { for (int m = 0; m < offset; ++m) acc[m] += acc[m + offset]; }
    }
    if (!part) y[size_t(slot) * H + row] = bf(acc[0]);
}
template<bool Down>
inline void launch_project(bool nv, Packed p, const int* ids, int fixed_expert, const float* input,
                           int stride, float* y, int slots) {
    if (nv) project<true,Down><<<dim3(H * 4 / 128, slots),128>>>(p, ids, fixed_expert, input, stride, y);
    else project<false,Down><<<dim3(H * 8 / 128, slots),128>>>(p, ids, fixed_expert, input, stride, y);
}

// One block row per slot. Rounds on read: batched GEMM rows arrive unrounded,
// and rounding rows that project() already rounded changes nothing.
static __global__ void activate(const float* gu, float* hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t slot = blockIdx.y;
    if (i >= F) return;
    // Original w13 stores alternating gate/up rows (HF Interleave(dim=1)).
    float g = bf(gu[slot * H + 2*i]), u = bf(gu[slot * H + 2*i+1]);
    float gate = tanhf(g / 4.f) * 4.f, up = tanhf(u / 25.f) * 25.f;
    hidden[slot * F + i] = bf(gate * (1.f / (1.f + expf(-g))) * up);
}

// One block row per token. rank maps (token,slot) to its down row when the
// rows were regrouped by expert; null means they are still in slot order.
static __global__ void combine(const float* down, const float* mix, const int* rank, float* out) {
    int token = blockIdx.y, row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;
    float v = 0.f;
    for (int j = 0; j < K; ++j) {
        int slot = token * K + j;
        v += __fmul_rn(bf(down[size_t(rank ? rank[slot] : slot) * H + row]), mix[slot]);
    }
    out[size_t(token) * H + row] = bf(v);
}

inline Packed load(Engine& e, const std::string& key, bool nv) {
    return {static_cast<const unsigned char*>(e.raw(key)),
            static_cast<const unsigned char*>(e.raw(key + "_scale")),
            nv ? e.w(key + "_amax") : nullptr,
            nv ? e.w(key + "_outer_scale_denominator") : nullptr};
}
} // namespace inkling_moe

inline void moe(Engine& e, int layer, const float* x, float* out) {
    using namespace inkling_moe;
    std::string key = "model.llm.layers." + std::to_string(layer) + ".mlp.";
    bool nv = layer < 20; // physical checkpoint layer, not loop visit
    auto w13 = load(e, key + "experts.w13_weight", nv), w2 = load(e, key + "experts.w2_weight", nv);
    static_assert(sizeof(int) == sizeof(float), "scratch capacity is in floats");
    auto* logits = e.tmp("moe_logits", N);
    auto* ids = reinterpret_cast<int*>(e.tmp("moe_ids", K));
    auto* mix = e.tmp("moe_mix", K);
    auto* projected = e.tmp("moe_projected", K * H);
    auto* hidden = e.tmp("moe_hidden", K * F);
    router<<<N / 4, 128>>>(e.w(key + "gate.weight"), x, logits);
    select<<<1, N>>>(logits, e.w(key + "gate.bias"), ids, mix);
    launch_project<false>(nv, w13, ids, 0, x, 0, projected, K);
    activate<<<dim3((F+127)/128,K),128>>>(projected, hidden);
    launch_project<true>(nv, w2, ids, 0, hidden, F, projected, K);
    combine<<<dim3((H+127)/128,1),128>>>(projected, mix, nullptr, out);
}
