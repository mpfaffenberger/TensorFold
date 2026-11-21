#pragma once
#include "model_ops.hpp"

namespace mini_prefill {
using mini_model::H;
using mini_model::T;
using mini_model::rank;
using mini_model::sum;
using mini_model::bf;

// Same per-lane fmaf sequence and shared reduction tree as scalar rms.
static __device__ void normalize(const float* x, const float* weight, float* y,
                                 bool round, float* scratch) {
    float ss = 0;
    for (int i=threadIdx.x; i<H; i+=T) ss = fmaf(x[i],x[i],ss);
    float scale = rsqrtf(sum(ss,scratch)/H + 1.e-6f);
    for (int i=threadIdx.x; i<H; i+=T) {
        float v = x[i]*scale*(weight ? weight[i] : 1.f);
        y[i] = round ? bf(v) : v;
    }
}
static __global__ void norm_rows(const float* x, const float* weight, float* y, bool round) {
    __shared__ float scratch[T];
    normalize(x+blockIdx.x*H,weight,y+blockIdx.x*H,round,scratch);
}
// Negative entries encode projected image row indices using bitwise complement.
static __global__ void embed_rows(const int* tokens, const float* image, const float* table,
                                  const float* weight, float* h) {
    __shared__ float scratch[T];
    int id = tokens[blockIdx.x];
    h += blockIdx.x*H;
    if (id >= 0) normalize(table+size_t(id)*H,weight,h,true,scratch);
    else for (int i=threadIdx.x; i<H; i+=T) h[i] = image[size_t(~id)*H+i];
}
// Planes are [state/query][row][H]. stride also lets the output head
// retrieve one row from the final chunk without compacting its state planes.
static __global__ void retrieve_rows(const float* states, int n, int stride,
                                     const float* queries, float* out) {
    __shared__ float scratch[T], scores[5];
    states += blockIdx.x*H;
    const float* q = queries+blockIdx.y*H;
    out += blockIdx.y*stride+blockIdx.x*H;
    for (int s=0; s<n; ++s) {
        float dot = 0;
        for (int i=threadIdx.x; i<H; i+=T) dot = fmaf(q[i],states[s*stride+i],dot);
        float logit = sum(dot,scratch)/48.f;
        if (!threadIdx.x) scores[s] = logit;
    }
    if (!threadIdx.x) {
        float peak = scores[0], total = 0;
        for (int s=1; s<n; ++s) peak = fmaxf(peak,scores[s]);
        for (int s=0; s<n; ++s) total += (scores[s]=expf(scores[s]-peak));
        for (int s=0; s<n; ++s) scores[s] /= total;
    }
    __syncthreads();
    for (int i=threadIdx.x; i<H; i+=T) {
        float v = 0;
        for (int s=0; s<n; ++s) v = fmaf(scores[s],states[s*stride+i],v);
        out[i] = v;
    }
}
// Normalize both halves without materializing unnormalized concatenations.
static __global__ void gate_input(const float* x, const float* msg, float* out) {
    __shared__ float scratch[T];
    normalize((blockIdx.y ? msg : x)+blockIdx.x*H,nullptr,
              out+(2*blockIdx.x+blockIdx.y)*H,true,scratch);
}
// forget and gain are the FP32 projections of the BF16-rounded low-rank state.
static __global__ void gate_mix(const float* forget, const float* gain, const float* x,
                                const float* msg, float* y) {
    int idx = blockIdx.x*T+threadIdx.x;
    y[idx] = (1.f-tanhf(forget[idx]))*x[idx] + tanhf(gain[idx])*msg[idx];
}
// x has a three-row prefix containing the scalar history (zero at start).
static __global__ void finish_rows(const float* x, const float* w, const float* residual, float* h) {
    int idx = blockIdx.x*T+threadIdx.x, i = idx%H;
    float a=x[idx], b=x[H+idx], c=x[2*H+idx], v=x[3*H+idx];
    float conv = a*w[4*i] + b*w[4*i+1] + c*w[4*i+2] + v*w[4*i+3];
    h[idx] = bf(residual[idx]+bf((v+conv)/sqrtf(5.f)));
}
} // namespace mini_prefill
