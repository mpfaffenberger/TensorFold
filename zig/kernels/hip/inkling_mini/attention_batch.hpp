#pragma once
#include "attention.hpp"
#include "checkpoint.hpp"
#include "dense.hpp"
#include <vector>

// Batched attention for prefill. Projections, Q.K scores and P.V mixes are
// GEMMs; rounding points, causal/sliding masks, sinks and convolutions follow
// the scalar kernels in attention.hpp. Sums are reassociated by the GEMMs, so
// this path is close to, not bit-identical with, scalar decode.
namespace inkling_attn_batch {
constexpr int Q = 3072, R = 384, KV = attn_kv, window = inkling_context::window;

// attn_norm for every row: one block per (head,row), blockDim.x lanes per head.
static __global__ void norm_heads(float* y, const float* scale, float clip, const float* taus, int width) {
    __shared__ float sums[128];
    int d = threadIdx.x, dim = blockDim.x, i = blockIdx.x*dim + d;
    float* row = y + size_t(blockIdx.y)*width;
    float a = row[i] * (scale ? scale[i] : 1.f);
    sums[d] = a*a;
    __syncthreads();
    for (int s = dim/2; s; s /= 2) {
        if (d < s) sums[d] += sums[d+s];
        __syncthreads();
    }
    float factor = clip ? fminf(1.f, clip / (sqrtf(sums[0]) + 1.e-6f))
                        : rsqrtf(sums[0]/dim + 1.e-6f);
    row[i] = attn_bf(attn_bf(a*factor)*(taus ? taus[blockIdx.y] : 1.f));
}

// attn_conv over a chunk. Taps read raw projections: earlier rows of this chunk
// when available, otherwise the four-slot ring left by the previous chunk.
static __global__ void conv_rows(const float* raw, float* out, const float* hist, const float* w,
                                 const float* dynamic, int n, int start) {
    int i = blockIdx.x*blockDim.x + threadIdx.x, row = blockIdx.y, pos = start+row;
    if (i >= n) return;
    float current = raw[size_t(row)*n+i], sum = 0.f;
    for (int j = 0; j < 4; ++j) {
        int lag = dynamic ? j : 3-j;
        if (pos < lag) continue;
        float tap = dynamic ? w[lag*n+i]+dynamic[row*32+(i/128)*4+lag] : w[i*4+j];
        sum += tap * (lag <= row ? raw[size_t(row-lag)*n+i] : hist[((pos-lag)%4)*n+i]);
    }
    out[size_t(row)*n+i] = attn_bf((sum+current)/(dynamic ? 3.f : sqrtf(5.f)));
}
// Publish up to four trailing raw rows so the next chunk or scalar decode continues.
static __global__ void hist_rows(const float* raw, float* hist, int n, int pos) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) hist[((pos+blockIdx.y)%4)*n+i] = raw[size_t(blockIdx.y)*n+i];
}

static __global__ void gather_rows(float** pages, int capacity, int first, float* keys, float* values) {
    int i = blockIdx.x*128 + threadIdx.x, t = first + blockIdx.y;
    const float* page = pages[(t/attn_page)%capacity];
    size_t at = size_t(blockIdx.y)*KV + i;
    keys[at] = attn_read(page,(t%attn_page)*KV+i);
    values[at] = attn_read(page,(attn_page+t%attn_page)*KV+i);
}
static __global__ void append_rows(float** pages, int capacity, int first, const float* keys, const float* values) {
    int i = blockIdx.x*128 + threadIdx.x, t = first + blockIdx.y;
    auto* bits = reinterpret_cast<uint16_t*>(pages[(t/attn_page)%capacity]);
    size_t at = size_t(blockIdx.y)*KV + i;
    bits[(t%attn_page)*KV+i] = __float_as_uint(keys[at]) >> 16;
    bits[(attn_page+t%attn_page)*KV+i] = __float_as_uint(values[at]) >> 16;
}

// One block per (head,query). `scores` enters holding raw Q.K dots for key
// positions [base,base+nk) and leaves holding probabilities, zero outside the
// query's causal window so the value GEMM can ignore the mask.
static __global__ void softmax_rows(float* scores, const float* bias, const float* sinks,
                                    int nk, int n, int base, int pos0, bool full) {
    __shared__ float buf[128];
    int h = blockIdx.x, qi = blockIdx.y, d = threadIdx.x, pos = pos0+qi;
    int first = full ? 0 : max(0,pos+1-window);
    float* s = scores + (size_t(h)*n + qi)*nk - base; // indexed by absolute key position
    const float* b = bias + (size_t(qi)*24 + h)*1024;
    float sink = sinks[qi*24+h], m = sink;
    for (int t = first+d; t <= pos; t += 128) {
        int lag = pos-t;
        s[t] = attn_bf(attn_bf(attn_bf(s[t])*0.08838834764831844f)+attn_bf(lag < 1024 ? b[lag] : 0.f));
        m = fmaxf(m,s[t]);
    }
    buf[d] = m;
    __syncthreads();
    for (int r = 64; r; r /= 2) {
        if (d < r) buf[d] = fmaxf(buf[d],buf[d+r]);
        __syncthreads();
    }
    m = buf[0];
    __syncthreads();
    float sum = d ? 0.f : expf(sink-m);
    for (int t = first+d; t <= pos; t += 128) { s[t] = expf(s[t]-m); sum += s[t]; }
    buf[d] = sum;
    __syncthreads();
    for (int r = 64; r; r /= 2) {
        if (d < r) buf[d] += buf[d+r];
        __syncthreads();
    }
    for (int t = first+d; t <= pos; t += 128) s[t] = attn_bf(s[t]/buf[0]);
    for (int t = base+d; t < base+nk; t += 128) if (t < first || t > pos) s[t] = 0.f;
}
static __global__ void gate_rows(float* mix, const float* gate, size_t count) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) mix[i] = attn_bf(attn_bf(mix[i])*(1.f/(1.f+expf(-gate[i]))));
}
} // namespace inkling_attn_batch

// q/k/v are B contiguous normalized rows each (they may alias). State is shared
// with scalar attention: continue either API at position start+B. `marks` are
// the checkpoints inside (start,start+B] whose rings to save on the way.
inline void attn_batch(Engine& e, int layer, int visit, int start, const float* q,
                       const float* k, const float* v, float* y, int B,
                       const std::vector<inkling_checkpoint::Mark>& marks = {}) {
    using namespace inkling_attn_batch;
    using mini_dense::gemm;
    if (layer < 0 || layer >= 24 || visit < 0 || visit >= 48 || start < 0 || B < 0 ||
        start > inkling_context::limit || B > inkling_context::limit-start)
        throw std::out_of_range("Inkling Mini batch attention range");
    if (!B) return;
    mini_engine::require(q && k && v && y,"null attention batch operand");

    const bool full = layer%6 == 5;
    const int capacity = inkling_context::capacity;
    const std::string base = "model.llm.layers."+std::to_string(layer)+".";
    const std::string state = inkling_context::state(layer,visit);
    auto w = [&](const std::string& key){ return e.w(base+"attn."+key); };
    auto tmp = [&](const std::string& key, size_t count){ return e.tmp("attn_batch."+key,count); };
    auto project = [&](const std::string& key, const float* x, const std::string& name, int out){
        float* rows = tmp(name,size_t(B)*out);
        mini_dense::linear_bf16(w(key),x,rows,B,2304,out);
        return rows;
    };
    const float* taus = nullptr; // log scaling depends on absolute position; host std::log matches decode
    if (full) {
        std::vector<float> host(B);
        for (int i = 0; i < B; ++i) host[i] = inkling_context::tau(true,start+i);
        float* device = tmp("tau",B);
        INKLING_HIP_CHECK(hipMemcpy(device,host.data(),B*sizeof(float),hipMemcpyHostToDevice));
        taus = device;
    }
    float* query = project("wq_du.weight",q,"query",Q);
    norm_heads<<<dim3(24,B),128>>>(query,w("q_l2_clip.scale"),22.627416998f,taus,Q);
    float* rel = project("wr_du.weight",q,"rel",R);
    norm_heads<<<dim3(24,B),16>>>(rel,w("rel_l2_clip.scale"),8.f,taus,R);
    float* sink = project("attn_sink_proj.weight",q,"sink",24);
    float* gate = project("gate_attn.weight",q,"gate",Q);

    // Contiguous float key window: cached positions the chunk can still see, then the chunk.
    const int past = full ? start : std::min(start,window-1), window_first = start-past;
    float* keys = tmp("keys",size_t(past+B)*KV), *values = tmp("values",size_t(past+B)*KV);
    static_assert(sizeof(float*)%sizeof(float)==0,"pointer table size");
    auto pages = reinterpret_cast<float**>(e.tmp(state+"directory",capacity*sizeof(float*)/sizeof(float)));
    if (past) gather_rows<<<dim3(8,past),128>>>(pages,capacity,window_first,keys,values);
    const int tail = std::min(B,4);
    // A ring as it stands after `boundary` positions: what it held before this
    // chunk, then the chunk's own rows up to the boundary. Call before the
    // live ring takes the chunk's tail.
    auto checkpoint = [&](const std::string& ring, const float* live, const float* raw, int width) {
        for (const auto& mark : marks) {
            float* saved = inkling_checkpoint::saved(e,ring,4*width,mark.slot);
            INKLING_HIP_CHECK(hipMemcpyAsync(saved,live,4*width*sizeof(float),hipMemcpyDeviceToDevice));
            const int from = std::max(start,mark.boundary-4);
            if (mark.boundary > from)
                hist_rows<<<dim3(width/128,mark.boundary-from),128>>>(raw+size_t(from-start)*width,saved,width,from);
        }
    };
    auto key_value = [&](bool is_key, const float* x, float* out){
        const std::string name = is_key ? "k" : "v";
        float* raw = project("w"+name+"_dv.weight",x,"kv_raw",KV);
        float* dynamic = full ? project(name+"_sconv.weight_proj",x,"kv_dynamic",32) : nullptr;
        float* hist = e.tmp(state+name+"hist",4*KV);
        conv_rows<<<dim3(8,B),128>>>(raw,out,hist,w(name+(full ? "_sconv.static_w" : "_sconv.weight")),dynamic,KV,start);
        checkpoint(state+name+"hist",hist,raw,KV);
        hist_rows<<<dim3(8,tail),128>>>(raw+size_t(B-tail)*KV,hist,KV,start+B-tail);
        norm_heads<<<dim3(8,B),128>>>(out,nullptr,is_key ? 22.627416998f : 0.f,nullptr,KV);
    };
    key_value(true,k,keys+size_t(past)*KV);
    key_value(false,v,values+size_t(past)*KV);

    // Bound the 24 x queries x keys score block near 800 MB however long the context is.
    int group = std::min(B,1024);
    if (full) group = std::min(group,std::max(32,int((size_t(8)<<20)/size_t(past+B))));
    const int widest = full ? past+B : std::min(past+B,window-1+group);
    float* scores = tmp("scores",size_t(24)*group*widest);
    float* bias = tmp("bias",size_t(group)*24*1024), *mix = tmp("mix",size_t(B)*Q);
    const float* table = w("rel_logits_proj.proj");
    for (int done = 0; done < B;) {
        const int n = std::min(group,B-done);
        const int k0 = full ? 0 : std::max(0,past+done-(window-1)), nk = past+done+n-k0;
        gemm(HIPBLAS_OP_N,HIPBLAS_OP_N,1024,n*24,16,table,1024,rel+size_t(done)*R,16,bias,1024);
        for (int h = 0; h < 24; ++h)
            gemm(HIPBLAS_OP_T,HIPBLAS_OP_N,nk,n,128,keys+size_t(k0)*KV+(h/3)*128,KV,
                 query+size_t(done)*Q+h*128,Q,scores+size_t(h)*n*nk,nk);
        softmax_rows<<<dim3(24,n),128>>>(scores,bias,sink+done*24,nk,n,window_first+k0,start+done,full);
        for (int h = 0; h < 24; ++h)
            gemm(HIPBLAS_OP_N,HIPBLAS_OP_N,128,n,nk,values+size_t(k0)*KV+(h/3)*128,KV,
                 scores+size_t(h)*n*nk,nk,mix+size_t(done)*Q+h*128,Q);
        done += n;
    }
    const size_t mixed = size_t(B)*Q;
    gate_rows<<<unsigned((mixed+255)/256),256>>>(mix,gate,mixed);
    float* projected = tmp("projected",size_t(B)*2304);
    mini_dense::linear_bf16(w("wo_ud.weight"),mix,projected,B,Q,2304);
    float* out_hist = e.tmp(state+"ohist",4*2304);
    conv_rows<<<dim3(18,B),128>>>(projected,y,out_hist,e.w(base+"attn_sconv.weight"),nullptr,2304,start);
    checkpoint(state+"ohist",out_hist,projected,2304);
    hist_rows<<<dim3(18,tail),128>>>(projected+size_t(B-tail)*2304,out_hist,2304,start+B-tail);

    // Publish the chunk to the paged cache.
    for (int index = start/attn_page; index <= (start+B-1)/attn_page; ++index) {
        float* page = e.tmp(state+"page."+std::to_string(index),inkling_context::page_floats);
        INKLING_HIP_CHECK(hipMemcpy(pages+index,&page,sizeof(page),hipMemcpyHostToDevice));
    }
    append_rows<<<dim3(8,B),128>>>(pages,capacity,start,keys+size_t(past)*KV,values+size_t(past)*KV);
    INKLING_HIP_CHECK(hipGetLastError());
}
