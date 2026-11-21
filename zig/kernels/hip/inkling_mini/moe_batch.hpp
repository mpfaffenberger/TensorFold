#pragma once
#include "moe.hpp"
#include "dense.hpp"

// Batched MoE for prefill. (token,slot) pairs are grouped by expert. A busy
// expert is dequantized once into FP32 scratch and applied with one GEMM per
// projection; an expert with only a few rows reads the packed weights directly,
// because dequantizing 8M weights for a handful of rows costs more than it saves.
// Routing, activation and mixing are the scalar kernels of moe.hpp over more rows.
namespace inkling_moe_batch {
using namespace inkling_moe;
// Rows per expert from which dequantize+GEMM is used. Prefill time was flat
// (within run-to-run noise) for 6..32 and slower at 48, on 150 to 5,000 token prompts.
constexpr int dense_rows = 12;

// One block, one thread per expert. order[r] is the pair at sorted rank r.
static __global__ void group_pairs(const int* ids, int pairs, int* counts, int* order, int* rank) {
    __shared__ int offsets[N];
    int expert = threadIdx.x, count = 0;
    for (int i = 0; i < pairs; ++i) count += ids[i] == expert;
    counts[expert] = offsets[expert] = count;
    __syncthreads();
    if (!expert) {
        int total = 0;
        for (int i = 0; i < N; ++i) { int own = offsets[i]; offsets[i] = total; total += own; }
    }
    __syncthreads();
    int at = offsets[expert];
    for (int i = 0; i < pairs; ++i)
        if (ids[i] == expert) { order[at] = i; rank[i] = at++; }
}
static __global__ void gather_rows(const float* x, const int* order, float* out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    out[size_t(blockIdx.y)*H+i] = x[size_t(order[blockIdx.y]/K)*H+i];
}

// One expert's BF16 weights as FP32, for a GEMM. One thread per 32-bit load.
template<bool NV>
static __global__ void dequantize(Packed p, int expert, int cols, float* out) {
    int local = (blockIdx.x*blockDim.x + threadIdx.x)*group<NV>;
    if (local < H*cols) unpack<NV>(p,expert*H*cols + local,outer_scale<NV>(p,expert),out+local);
}

template<bool Down>
inline void apply_experts(Packed p, bool nv, const int* counts, const float* in, float* out, float* weight) {
    constexpr int cols = Down ? F : H;
    for (int expert = 0, at = 0; expert < N; at += counts[expert++]) {
        const int count = counts[expert];
        if (!count) continue;
        const float* rows = in + size_t(at)*cols;
        float* result = out + size_t(at)*H;
        if (count < dense_rows) {
            launch_project<Down>(nv,p,nullptr,expert,rows,cols,result,count);
        } else {
            const unsigned loads = H*cols/(nv ? group<true> : group<false>);
            if (nv) dequantize<true><<<loads/256,256>>>(p,expert,cols,weight);
            else dequantize<false><<<loads/256,256>>>(p,expert,cols,weight);
            mini_dense::linear(weight,rows,result,count,cols,H);
        }
    }
}
} // namespace inkling_moe_batch

// x and y are B contiguous rows. Same routing, rounding points and expert
// mixing order as moe(); dot products inside a GEMM are reassociated.
inline void moe_batch(Engine& e, int layer, const float* x, float* y, int B) {
    using namespace inkling_moe_batch;
    if (B <= 0) return;
    const std::string key = "model.llm.layers." + std::to_string(layer) + ".mlp.";
    const bool nv = layer < 20; // physical checkpoint layer, not loop visit
    auto w13 = load(e,key+"experts.w13_weight",nv), w2 = load(e,key+"experts.w2_weight",nv);
    const int pairs = B*K;
    auto tmp = [&](const char* name, size_t count){ return e.tmp(std::string("moe_batch.")+name,count); };
    static_assert(sizeof(int) == sizeof(float), "scratch capacity is in floats");
    auto ints = [&](const char* name, size_t count){ return reinterpret_cast<int*>(tmp(name,count)); };
    float* logits = tmp("logits",size_t(B)*N), *mix = tmp("mix",pairs);
    int* ids = ints("ids",pairs), *order = ints("order",pairs), *rank = ints("rank",pairs), *counts = ints("counts",N);
    float* inputs = tmp("inputs",size_t(pairs)*H), *projected = tmp("projected",size_t(pairs)*H);
    float* hidden = tmp("hidden",size_t(pairs)*F), *weight = tmp("weight",size_t(H)*H);

    mini_dense::linear(e.w(key+"gate.weight"),x,logits,B,H,N); // the router stays FP32, unrounded
    select<<<B,N>>>(logits,e.w(key+"gate.bias"),ids,mix);
    group_pairs<<<1,N>>>(ids,pairs,counts,order,rank);
    gather_rows<<<dim3(H/256,pairs),256>>>(x,order,inputs);
    int host_counts[N];
    INKLING_HIP_CHECK(hipMemcpy(host_counts,counts,sizeof(host_counts),hipMemcpyDeviceToHost));
    apply_experts<false>(w13,nv,host_counts,inputs,projected,weight);
    activate<<<dim3(F/128,pairs),128>>>(projected,hidden);
    apply_experts<true>(w2,nv,host_counts,hidden,projected,weight); // `projected` now holds down rows
    combine<<<dim3(H/128,B),128>>>(projected,mix,rank,y);
    INKLING_HIP_CHECK(hipGetLastError());
}
