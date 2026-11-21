#pragma once
#include "attention.hpp"

namespace inkling_attn_batch {
// One block owns a weight row and a tile of tokens. Keep Engine::mv's 256
// independent FMA chains and reduction tree for each token, but load a weight
// only once for the whole tile. Invalid tail tokens never read input.
constexpr int tile = 4;
static __global__ void project_rows(const float* weight, const float* x, float* y,
                                    int rows, int cols, int count) {
    constexpr int lanes = 256;
    __shared__ float sums[tile][lanes];
    int lane = threadIdx.x, row = blockIdx.x, first = blockIdx.y * tile;
    float acc[tile] = {};
    for (int col = lane; col < cols; col += lanes) {
        float w = weight[row*cols+col];
        #pragma unroll
        for (int j = 0; j < tile; ++j)
            if (first+j < count) acc[j] = fmaf(w,x[(first+j)*cols+col],acc[j]);
    }
    for (int j = 0; j < tile; ++j) sums[j][lane] = acc[j];
    __syncthreads();
    for (int stride = lanes/2; stride; stride /= 2) {
        if (lane < stride) for (int j = 0; j < tile; ++j) sums[j][lane] += sums[j][lane+stride];
        __syncthreads();
    }
    if (lane < tile && first+lane < count) {
        // Match Engine::mv's NaN-preserving BF16 conversion.
        unsigned bits = __float_as_uint(sums[lane][0]);
        if ((bits & 0x7fffffff) > 0x7f800000) bits |= 0x00400000;
        else bits += 0x7fff + ((bits >> 16) & 1);
        y[(first+lane)*rows+row] = __uint_as_float(bits & 0xffff0000);
    }
}
} // namespace inkling_attn_batch

// One visit, B consecutive positions. Each input and the output is a contiguous
// [B,2304] device tensor; Q/K/V are already normalized as for attn_forward_qkv.
// State is shared with scalar attention: continue either API at start+B.
inline void attn_batch(Engine& e, int layer, int visit, int start, const float* q,
                       const float* k, const float* v, float* y, int B) {
    if (layer < 0 || layer >= 24 || visit < 0 || visit >= 48 || start < 0 ||
        start > inkling_context::limit || B < 0 || B > inkling_context::limit-start)
        throw std::out_of_range("Inkling Mini batch attention range");
    if (!B) return;
    mini_engine::require(q && k && v && y,"null attention batch operand");

    const bool full = layer%6 == 5;
    const int capacity = inkling_context::capacity(full);
    const std::string base = "model.llm.layers."+std::to_string(layer)+".";
    const std::string state = inkling_context::state(layer,visit);
    auto w = [&](const char* key){ return e.w(base+"attn."+key); };
    const float *qs = w("q_l2_clip.scale"), *rs = w("rel_l2_clip.scale");
    const float *kc = w(full ? "k_sconv.static_w" : "k_sconv.weight");
    const float *vc = w(full ? "v_sconv.static_w" : "v_sconv.weight");
    const float *table = w("rel_logits_proj.proj"), *oc = e.w(base+"attn_sconv.weight");
    float *kh = e.tmp(state+"khist",4096), *vh = e.tmp(state+"vhist",4096);
    float* oh = e.tmp(state+"ohist",4*2304);

    // Bound transient storage independently of prompt length. Only the dense
    // projections cross token boundaries; KV pages and conv rings advance in
    // default-stream order using the scalar kernels and absolute positions.
    constexpr int chunk = 64;
    for (int offset = 0; offset < B; offset += chunk) {
        int n = std::min(chunk,B-offset);
        auto tmp = [&](const char* key, int width){
            return e.tmp(std::string("attn_batch_work.")+key,n*width);
        };
        auto project = [&](const char* key, const float* x, int rows, int cols = 2304){
            float* out = tmp(key,rows);
            constexpr int tile = inkling_attn_batch::tile;
            inkling_attn_batch::project_rows<<<dim3(rows,(n+tile-1)/tile),256>>>(w(key),x,out,rows,cols,n);
            INKLING_HIP_CHECK(hipGetLastError());
            return out;
        };
        const float *xq = q+offset*2304, *xk = k+offset*2304, *xv = v+offset*2304;
        float *query = project("wq_du.weight",xq,3072), *rel = project("wr_du.weight",xq,384);
        float *keys = project("wk_dv.weight",xk,1024), *values = project("wv_dv.weight",xv,1024);
        float* kd = full ? project("k_sconv.weight_proj",xk,32) : nullptr;
        float* vd = full ? project("v_sconv.weight_proj",xv,32) : nullptr;
        float *sink = project("attn_sink_proj.weight",xq,24), *gate = project("gate_attn.weight",xq,3072);
        float* mix = tmp("mix",3072);
        for (int i = 0; i < n; ++i) {
            int pos = start+offset+i, first = inkling_context::first(full,pos);
            float tau = inkling_context::tau(full,pos);
            float *qi = query+i*3072, *ri = rel+i*384, *ki = keys+i*1024, *vi = values+i*1024;
            attn_norm<<<24,128>>>(qi,qs,22.627416998f,tau);
            attn_norm<<<24,16>>>(ri,rs,8.f,tau);
            attn_conv<<<8,128>>>(ki,ki,kh,kc,full ? kd+i*32 : nullptr,1024,pos);
            attn_norm<<<8,128>>>(ki,nullptr,22.627416998f,1.f);
            attn_conv<<<8,128>>>(vi,vi,vh,vc,full ? vd+i*32 : nullptr,1024,pos);
            attn_norm<<<8,128>>>(vi,nullptr,0.f,1.f);
            auto pages = attn_cache_append(e,state,capacity,pos,ki,vi);
            attn_scores<<<dim3((pos-first+4)/4,24),128>>>(pages,capacity,first,pos,qi,ri,table);
            attn_softmax<<<24,128>>>(pages,capacity,first,pos,sink+i*24);
            attn_mix<<<24,128>>>(pages,capacity,first,pos,gate+i*3072,mix+i*3072);
        }
        float* projected = project("wo_ud.weight",mix,2304,3072);
        for (int i = 0; i < n; ++i)
            attn_conv<<<18,128>>>(projected+i*2304,y+(offset+i)*2304,oh,oc,nullptr,2304,start+offset+i);
        INKLING_HIP_CHECK(hipGetLastError());
    }
}
