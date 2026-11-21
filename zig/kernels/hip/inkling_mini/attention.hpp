#pragma once
#include "engine.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Inkling-2-Mini, one unpadded token, default HIP stream (including Engine::mv).
// layer is the physical checkpoint layer, visit distinguishes uses of that layer
// within one token. Use the same visit IDs at every position. pos starts at 0;
// restarting at 0 logically resets all history. One sequential decoder per Engine.
// x packs three contiguous normalized, BF16-rounded 2304-float vectors [Q,K,V]. Output INCLUDES attn_sconv and does
// NOT include the residual, block mixing, layer norms, message gates or MLP.
// Use attn_forward_qkv for noncontiguous inputs; pass x,x,x there when shared.
// Sink, relative and output gate use Q input.

static __device__ inline float attn_bf(float v) {
    unsigned u = __float_as_uint(v);
    return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}

// One block per head, with exactly dim lanes (16 or 128).
static __global__ void attn_norm(float* y, const float* scale, float clip, float tau) {
    __shared__ float sums[128];
    int d = threadIdx.x, dim = blockDim.x, i = blockIdx.x * dim + d;
    float a = y[i] * (scale ? scale[i] : 1.f);
    sums[d] = a*a;
    __syncthreads();
    for (int s = dim/2; s; s /= 2) {
        if (d < s) sums[d] += sums[d+s];
        __syncthreads();
    }
    float factor = clip ? fminf(1.f, clip / (sqrtf(sums[0]) + 1.e-6f))
                        : rsqrtf(sums[0]/dim + 1.e-6f);
    y[i] = attn_bf(attn_bf(a*factor)*tau);
}

// hist holds raw projection outputs, not filtered outputs. Static conv1d's
// last column is the current token; dynamic static_w's first ROW is current.
static __global__ void attn_conv(const float* x, float* out, float* hist, const float* w,
                                 const float* dynamic, int n, int pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float current = x[i], sum = 0.f;
    hist[(pos%4)*n+i] = current;
    for (int j = 0; j < 4; ++j) {
        int lag = dynamic ? j : 3-j;
        if (pos < lag) continue;
        float tap = dynamic ? w[lag*n+i]+dynamic[(i/128)*4+lag] : w[i*4+j];
        sum += tap * hist[((pos-lag)%4)*n+i];
    }
    out[i] = attn_bf((sum+current)/(dynamic ? 3.f : sqrtf(5.f)));
}

// Fixed-size pages avoid allocating model_max_length worth of KV at startup,
// or relying on tmp() preserving data when resized. Last segment is per-head
// score scratch. A sliding visit reuses 16 pages in a ring; full visits never wrap.
static constexpr int attn_page = 64, attn_kv = 1024, attn_heads = 24;
static __global__ void attn_append(float** pages, int index, float* page, int slot,
                                   const float* k, const float* v) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (!i) pages[index] = page;
    page[slot*attn_kv+i] = k[i];
    page[(attn_page+slot)*attn_kv+i] = v[i];
}

static __device__ inline float* attn_score_ptr(float* page, int h, int t) {
    return page + 2*attn_page*attn_kv + h*attn_page + t%attn_page;
}

static __global__ void attn_scores(float** pages, int capacity, int first, int pos,
                                   const float* q, const float* rel, const float* table) {
    __shared__ float dot[128];
    int t = first+blockIdx.x, h = blockIdx.y, d = threadIdx.x, lag = pos-t;
    float* page = pages[(t/attn_page)%capacity];
    dot[d] = q[h*128+d]*page[(t%attn_page)*attn_kv+(h/3)*128+d];
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (d < s) dot[d] += dot[d+s];
        __syncthreads();
    }
    if (!d) {
        float bias = 0.f;
        if (lag < 1024) for (int r = 0; r < 16; ++r) bias += rel[h*16+r]*table[r*1024+lag];
        *attn_score_ptr(page,h,t) = attn_bf(attn_bf(attn_bf(dot[0])*0.08838834764831844f)+attn_bf(bias));
    }
}

static __global__ void attn_softmax(float** pages, int capacity, int first, int pos, const float* sinks) {
    __shared__ float buf[128];
    int h = blockIdx.x, d = threadIdx.x;
    float m = sinks[h];
    for (int t = first+d; t <= pos; t += 128)
        m = fmaxf(m,*attn_score_ptr(pages[(t/attn_page)%capacity],h,t));
    buf[d] = m;
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (d < s) buf[d] = fmaxf(buf[d],buf[d+s]);
        __syncthreads();
    }
    m = buf[0];
    __syncthreads();
    float sum = d ? 0.f : expf(sinks[h]-m);
    for (int t = first+d; t <= pos; t += 128) {
        float* p = attn_score_ptr(pages[(t/attn_page)%capacity],h,t);
        *p = expf(*p-m); sum += *p;
    }
    buf[d] = sum;
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (d < s) buf[d] += buf[d+s];
        __syncthreads();
    }
    for (int t = first+d; t <= pos; t += 128) {
        float* p = attn_score_ptr(pages[(t/attn_page)%capacity],h,t);
        *p = attn_bf(*p/buf[0]);
    }
}

static __global__ void attn_mix(float** pages, int capacity, int first, int pos,
                                const float* gate, float* y) {
    int h = blockIdx.x, d = threadIdx.x, i = h*128+d;
    float sum = 0.f;
    for (int t = first; t <= pos; ++t) {
        float* page = pages[(t/attn_page)%capacity];
        sum += *attn_score_ptr(page,h,t)*page[(attn_page+t%attn_page)*attn_kv+(h/3)*128+d];
    }
    y[i] = attn_bf(attn_bf(sum)*(1.f/(1.f+expf(-gate[i]))));
}

inline void attn_forward_qkv(Engine& e, int layer, int visit, int pos, const float* x,
                             const float* xk, const float* xv, float* out) {
    if (layer < 0 || layer >= 24 || visit < 0 || pos < 0 || pos >= 1048576)
        throw std::out_of_range("Inkling Mini attention index");
    const bool full = layer%6 == 5;
    const std::string base = "model.llm.layers."+std::to_string(layer)+".";
    const std::string state = "attn_state."+std::to_string(layer)+"."+std::to_string(visit)+".";
    auto w = [&](const std::string& key){ return e.w(base+"attn."+key); };
    auto tmp = [&](const std::string& key, size_t n){ return e.tmp("attn_work."+key,n); };
    const float tau = full ? 1.f+.1f*std::log(std::max(1.f,(pos+1)/8192.f)) : 1.f;
    float* q = tmp("q",3072), *rel = tmp("rel",384);
    e.mv(w("wq_du.weight"),x,q,3072,2304);
    attn_norm<<<24,128>>>(q,w("q_l2_clip.scale"),22.627416998f,tau);
    e.mv(w("wr_du.weight"),x,rel,384,2304);
    attn_norm<<<24,16>>>(rel,w("rel_l2_clip.scale"),8.f,tau);

    auto kv = [&](bool key, const float* input){
        std::string name = key ? "k" : "v";
        float* y = tmp(name,1024);
        e.mv(w("w"+name+"_dv.weight"),input,y,1024,2304);
        float* dyn = nullptr;
        if (full) {
            dyn = tmp("dynamic",32);
            e.mv(w(name+"_sconv.weight_proj"),input,dyn,32,2304);
        }
        attn_conv<<<8,128>>>(y,y,e.tmp(state+name+"hist",4096),
                             w(name+(full ? "_sconv.static_w" : "_sconv.weight")),dyn,1024,pos);
        attn_norm<<<8,128>>>(y,nullptr,key ? 22.627416998f : 0.f,1.f);
        return y;
    };
    float* k = kv(true,xk), *v = kv(false,xv);
    int capacity = (full ? 1048576 : 1024)/attn_page, index = (pos/attn_page)%capacity;
    static_assert(sizeof(float*)%sizeof(float)==0,"pointer table size");
    auto pages = reinterpret_cast<float**>(e.tmp(state+"directory",capacity*sizeof(float*)/sizeof(float)));
    float* page = e.tmp(state+"page."+std::to_string(index),attn_page*(2*attn_kv+attn_heads));
    attn_append<<<8,128>>>(pages,index,page,pos%attn_page,k,v);
    int first = full ? 0 : std::max(0,pos-1023);
    float* sink = tmp("sink",24), *gate = tmp("gate",3072), *mix = tmp("mix",3072);
    e.mv(w("attn_sink_proj.weight"),x,sink,24,2304);
    e.mv(w("gate_attn.weight"),x,gate,3072,2304);
    attn_scores<<<dim3(pos-first+1,24),128>>>(pages,capacity,first,pos,q,rel,w("rel_logits_proj.proj"));
    attn_softmax<<<24,128>>>(pages,capacity,first,pos,sink);
    attn_mix<<<24,128>>>(pages,capacity,first,pos,gate,mix);
    float* projected = tmp("projected",2304);
    e.mv(w("wo_ud.weight"),mix,projected,2304,3072);
    attn_conv<<<18,128>>>(projected,out,e.tmp(state+"ohist",4*2304),e.w(base+"attn_sconv.weight"),nullptr,2304,pos);
}

inline void attention(Engine& e, int layer, int visit, int pos, const float* x, float* out) {
    attn_forward_qkv(e,layer,visit,pos,x,x+2304,x+4608,out);
}
