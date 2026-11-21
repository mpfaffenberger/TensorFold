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

static __global__ void router(const float* w, const float* x, float* scores) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    float v = 0.f;
    for (int c = lane; c < H; c += 32) v = fmaf(w[row * H + c], x[c], v);
    v = sum32(v); // The router's linear is FP32, not a BF16-output mv.
    if (!lane) scores[row] = 1.f / (1.f + expf(-v));
}

static __global__ void select(const float* scores, const float* bias, float* ids, float* mix) {
    int best[K];
    float ranks[K];
    for (int j = 0; j < K; ++j) { ranks[j] = -INFINITY; best[j] = 0; }
    for (int i = 0; i < N; ++i) {
        float rank = scores[i] + bias[i];
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
        ids[pos] = float(best[j]);
        mix[pos] = (scores[best[j]] / (total + 1e-20f)) * 8.f;
    }
}

struct Packed {
    const unsigned char *w, *scale;
    const float *amax, *denom;
};

template<bool NV, bool Down>
static __global__ void project(Packed p, const float* ids, const float* input, float* y) {
    constexpr int cols = Down ? F : H;
    int slot = blockIdx.y, expert = int(ids[slot]);
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    int base = (expert * H + row) * cols; // Both projections have H output rows (2*F == H).
    const float* x = input + (Down ? slot * F : 0);
    float factor = 1.f, sum = 0.f;
    // The pinned loader materializes all converter inputs in the target BF16
    // dtype before Inkling2LoopDequantizeExperts does its FP32 arithmetic.
    if (NV) factor = bf(p.amax[expert]) / bf(p.denom[0]);
    for (int c = lane; c < cols; c += 32) {
        int i = base + c;
        float w;
        if (NV) {
            unsigned byte = p.w[i / 2];
            w = e2m1((byte >> ((i & 1) * 4)) & 15) * e4m3(p.scale[i / 16]);
            w *= factor;
        } else {
            w = e4m3(p.w[i]) * exp2f(float(p.scale[i / 32]) - 127.f);
        }
        sum = fmaf(bf(w), x[c], sum);
    }
    sum = sum32(sum);
    if (!lane) y[slot * H + row] = bf(sum);
}

static __global__ void activate(const float* gu, float* hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, slot = blockIdx.y;
    if (i >= F) return;
    // Original w13 stores alternating gate/up rows (HF Interleave(dim=1)).
    float g = gu[slot * H + 2*i], u = gu[slot * H + 2*i+1];
    float gate = tanhf(g / 4.f) * 4.f, up = tanhf(u / 25.f) * 25.f;
    hidden[slot * F + i] = bf(gate * (1.f / (1.f + expf(-g))) * up);
}

static __global__ void combine(const float* down, const float* mix, float* out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;
    float v = 0.f;
    for (int j = 0; j < K; ++j) v += __fmul_rn(down[j * H + row], mix[j]);
    out[row] = bf(v);
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
    auto* scores = e.tmp("moe_scores", N);
    auto* ids = e.tmp("moe_ids", K);
    auto* mix = e.tmp("moe_mix", K);
    auto* projected = e.tmp("moe_projected", K * H);
    auto* hidden = e.tmp("moe_hidden", K * F);
    router<<<N / 4, 128>>>(e.w(key + "gate.weight"), x, scores);
    select<<<1, 1>>>(scores, e.w(key + "gate.bias"), ids, mix);
    dim3 grid(H / 4, K);
    if (nv) project<true,false><<<grid,128>>>(w13, ids, x, projected);
    else project<false,false><<<grid,128>>>(w13, ids, x, projected);
    activate<<<dim3((F+127)/128,K),128>>>(projected, hidden);
    if (nv) project<true,true><<<grid,128>>>(w2, ids, hidden, projected);
    else project<false,true><<<grid,128>>>(w2, ids, hidden, projected);
    combine<<<(H+127)/128,128>>>(projected, mix, out);
}
