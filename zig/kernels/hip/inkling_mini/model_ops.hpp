#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mini_model {
constexpr int H = 2304, T = 256, rank = 64, vocab = 200058;
constexpr int blocks = H / T;
static_assert(blocks * T == H, "elementwise kernels cover exactly H");

static __device__ float bf(float x) {
    unsigned u = __float_as_uint(x);
    return __uint_as_float((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000);
}
static __device__ float sum(float x, float* shared) {
    int t = threadIdx.x;
    shared[t] = x;
    __syncthreads();
    for (int stride = T/2; stride; stride /= 2) {
        if (t < stride) shared[t] += shared[t + stride];
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}
static __global__ void rms(const float* x, const float* weight, float* y, bool round) {
    __shared__ float scratch[T];
    float ss = 0;
    for (int i = threadIdx.x; i < H; i += T) ss = fmaf(x[i], x[i], ss);
    float scale = rsqrtf(sum(ss, scratch) / H + 1.e-6f);
    for (int i = threadIdx.x; i < H; i += T) {
        float v = x[i] * scale * (weight ? weight[i] : 1.f);
        y[i] = round ? bf(v) : v;
    }
}
// Both keys and values are normalized. Up to four completed blocks + partial.
static __global__ void retrieve(const float* states, int n, const float* queries, float* out) {
    __shared__ float scratch[T], scores[5];
    const float* q = queries + blockIdx.x * H;
    for (int s = 0; s < n; ++s) {
        float dot = 0;
        for (int i = threadIdx.x; i < H; i += T) dot = fmaf(q[i], states[s*H+i], dot);
        float logit = sum(dot, scratch) / 48.f; // sqrt(2304)
        if (!threadIdx.x) scores[s] = logit;
    }
    if (!threadIdx.x) {
        float peak = scores[0], total = 0;
        for (int s = 1; s < n; ++s) peak = fmaxf(peak, scores[s]);
        for (int s = 0; s < n; ++s) total += (scores[s] = expf(scores[s]-peak));
        for (int s = 0; s < n; ++s) scores[s] /= total;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < H; i += T) {
        float v = 0;
        for (int s = 0; s < n; ++s) v = fmaf(scores[s], states[s*H+i], v);
        out[blockIdx.x*H+i] = v;
    }
}
// HF keeps gain and forget projections in FP32; e.mv rounds its output to BF16.
static __global__ void gate_up(const float* low, const float* forget, const float* gain,
                               const float* x, const float* msg, float* y) {
    int i = blockIdx.x * T + threadIdx.x;
    float f = 0, g = 0;
    for (int j = 0; j < rank; ++j) {
        f = fmaf(forget[i*rank+j], low[j], f);
        g = fmaf(gain[i*rank+j], low[j], g);
    }
    y[i] = (1.f - tanhf(f))*x[i] + tanhf(g)*msg[i];
}
static __global__ void add(const float* x, const float* y, float* out) {
    int i = blockIdx.x*T + threadIdx.x;
    out[i] = x[i] + y[i];
}
// History stores the three preceding MoE outputs, independently for every visit.
static __global__ void finish_mlp(const float* x, const float* w, float* hist, int pos,
                                  const float* residual, float* out) {
    int i = blockIdx.x*T + threadIdx.x;
    float a = pos >= 3 ? hist[i] : 0.f;
    float b = pos >= 2 ? hist[H+i] : 0.f;
    float c = pos >= 1 ? hist[2*H+i] : 0.f;
    float v = x[i];
    hist[i] = b; hist[H+i] = c; hist[2*H+i] = v;
    float conv = a*w[4*i] + b*w[4*i+1] + c*w[4*i+2] + v*w[4*i+3];
    out[i] = bf(residual[i] + bf((v+conv) / sqrtf(5.f)));
}
static __global__ void greedy(float* logits, int* id) {
    __shared__ float values[T];
    __shared__ int ids[T];
    int t = threadIdx.x, best = t;
    float peak = -INFINITY;
    for (int i = t; i < vocab; i += T) {
        float v = logits[i] = 30.f * tanhf(logits[i] / 30.f);
        if (v > peak) { peak = v; best = i; }
    }
    values[t] = peak; ids[t] = best;
    __syncthreads();
    for (int s = T/2; s; s /= 2) {
        if (t < s && (values[t+s] > values[t] ||
            (values[t+s] == values[t] && ids[t+s] < ids[t]))) {
            values[t] = values[t+s]; ids[t] = ids[t+s];
        }
        __syncthreads();
    }
    if (!t) *id = ids[0];
}
inline void check(hipError_t err) {
    if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}
inline void copy(float* dst, const float* src, int n) {
    check(hipMemcpyAsync(dst, src, n*sizeof(float), hipMemcpyDeviceToDevice));
}
} // namespace mini_model
