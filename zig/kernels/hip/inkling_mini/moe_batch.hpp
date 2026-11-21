#pragma once
#include "moe.hpp"

namespace inkling_moe_batch {
using namespace inkling_moe;
constexpr int Tile = 4, Chunk = 128;

static __global__ void route(const float* w, const float* x, float* scores) {
    int t = blockIdx.y, lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    float v = 0.f;
    for (int c = lane; c < H; c += 32) v = fmaf(w[row*H+c], x[t*H+c], v);
    v = sum32(v);
    if (!lane) scores[t*N+row] = 1.f / (1.f + expf(-v));
}

static __global__ void choose(const float* scores, const float* bias, float* ids, float* mix) {
    int t = blockIdx.x;
    scores += t*N;
    int best[K] = {};
    float ranks[K];
    for (int j = 0; j < K; ++j) ranks[j] = -INFINITY;
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
    for (int j = 0; j < K; ++j) {
        int pos = 0;
        for (int i = 0; i < K; ++i) pos += best[i] < best[j];
        ids[t*K+pos] = float(best[j]);
        mix[t*K+pos] = (scores[best[j]] / (total + 1e-20f)) * 8.f;
    }
}

// One block constructs a compact tile schedule on device. Routing selects an
// expert at most once per token, so its list needs at most B original slots.
static __global__ void schedule(const float* ids, int B, int* slots, int* tasks) {
    __shared__ int offsets[N+1];
    int e = threadIdx.x, count = 0;
    for (int i = 0; i < B*K; ++i)
        if (int(ids[i]) == e) slots[e*B + count++] = i;
    offsets[e+1] = (count+Tile-1)/Tile;
    __syncthreads();
    if (!e) {
        offsets[0] = 0;
        for (int i = 0; i < N; ++i) offsets[i+1] += offsets[i];
        tasks[0] = offsets[N];
    }
    __syncthreads();
    for (int i = 0; i < count; i += Tile) {
        int* task = tasks + 1 + 3*(offsets[e]+i/Tile);
        task[0] = e; task[1] = e*B+i; task[2] = min(Tile,count-i);
    }
}

template<bool NV, bool Down>
static __global__ void grouped(Packed p, const int* tasks, const int* slots,
                               const float* input, float* output) {
    if (blockIdx.y >= tasks[0]) return;
    const int* task = tasks + 1 + 3*blockIdx.y;
    int expert = task[0], n = task[2];
    int row = blockIdx.x*4 + threadIdx.x/32, lane = threadIdx.x%32;
    constexpr int cols = Down ? F : H;
    int base = (expert*H + row)*cols;
    float factor = NV ? bf(p.amax[expert]) / bf(p.denom[0]) : 1.f;
    int slot[Tile] = {};
    float sum[Tile] = {};
    #pragma unroll
    for (int j = 0; j < Tile; ++j)
        if (j < n) slot[j] = slots[task[1]+j];
    for (int c = lane; c < cols; c += 32) {
        int i = base+c;
        float w;
        if (NV) {
            unsigned byte = p.w[i/2];
            w = e2m1((byte >> ((i&1)*4)) & 15) * e4m3(p.scale[i/16]);
            w *= factor;
        } else {
            w = e4m3(p.w[i]) * exp2f(float(p.scale[i/32])-127.f);
        }
        w = bf(w);
        #pragma unroll
        for (int j = 0; j < Tile; ++j) if (j < n) {
            int src = Down ? slot[j] : slot[j]/K;
            sum[j] = fmaf(w, input[src*cols+c], sum[j]);
        }
    }
    #pragma unroll
    for (int j = 0; j < Tile; ++j) if (j < n) {
        float v = sum32(sum[j]);
        if (!lane) output[slot[j]*H+row] = bf(v);
    }
}

static __global__ void finish(const float* down, const float* mix, float* out) {
    int t = blockIdx.y, row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= H) return;
    float v = 0.f;
    for (int j = 0; j < K; ++j) v += __fmul_rn(down[(t*K+j)*H+row], mix[t*K+j]);
    out[t*H+row] = bf(v);
}
} // namespace inkling_moe_batch

inline void moe_batch(Engine& e, int layer, const float* x, float* y, int B) {
    using namespace inkling_moe_batch;
    if (B <= 0) return;
    if (B == 1) return moe(e, layer, x, y);
    std::string key = "model.llm.layers." + std::to_string(layer) + ".mlp.";
    bool nv = layer < 20;
    auto w13 = load(e,key+"experts.w13_weight",nv), w2 = load(e,key+"experts.w2_weight",nv);
    auto gate = e.w(key+"gate.weight"), bias = e.w(key+"gate.bias");
    int cap = std::min(B,Chunk);
    auto scores = e.tmp("moeb_scores",cap*N), ids = e.tmp("moeb_ids",cap*K);
    auto mix = e.tmp("moeb_mix",cap*K), gu = e.tmp("moeb_projected",cap*K*H);
    auto hidden = e.tmp("moeb_hidden",cap*K*F);
    static_assert(sizeof(int) == sizeof(float), "scratch capacity is in floats");
    auto slots = reinterpret_cast<int*>(e.tmp("moeb_slots",cap*N));
    auto tasks = reinterpret_cast<int*>(e.tmp("moeb_tasks",1+3*cap*K));
    while (B) {
        int b = std::min(B,cap);
        route<<<dim3(N/4,b),128>>>(gate,x,scores);
        choose<<<b,1>>>(scores,bias,ids,mix);
        schedule<<<1,N>>>(ids,b,slots,tasks);
        // At most Tile-1 padding entries per expert; no host count readback.
        dim3 grid(H/4,std::min(b*K,(b*K+N*(Tile-1))/Tile));
        if (nv) grouped<true,false><<<grid,128>>>(w13,tasks,slots,x,gu);
        else grouped<false,false><<<grid,128>>>(w13,tasks,slots,x,gu);
        activate<<<dim3((F+127)/128,b*K),128>>>(gu,hidden);
        if (nv) grouped<true,true><<<grid,128>>>(w2,tasks,slots,hidden,gu);
        else grouped<false,true><<<grid,128>>>(w2,tasks,slots,hidden,gu);
        finish<<<dim3((H+127)/128,b),128>>>(gu,mix,y);
        x += b*H; y += b*H; B -= b;
    }
}
