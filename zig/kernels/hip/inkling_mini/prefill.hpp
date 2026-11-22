#pragma once
#include "attention_batch.hpp"
#include "moe_batch.hpp"
#include "prefill_ops.hpp"
#include "dense.hpp"
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstdlib>

namespace mini_prefill {
// TF_INKLING_PROFILE=1 attributes prefill time to stages. It synchronizes the
// device at every stage boundary, so the profiled run is slower than a normal one.
struct Stages {
    enum Stage { glue, attention, experts, count };
    using Clock = std::chrono::steady_clock;
    bool enabled = std::getenv("TF_INKLING_PROFILE") != nullptr;
    double seconds[count] = {};
    Clock::time_point mark = Clock::now();
    void lap(Stage stage) {
        if (!enabled) return;
        INKLING_HIP_CHECK(hipDeviceSynchronize());
        auto now = Clock::now();
        seconds[stage] += std::chrono::duration<double>(now-mark).count();
        mark = now;
    }
    ~Stages() {
        if (enabled) std::fprintf(stderr,"Inkling prefill stages: attention %.3fs, experts %.3fs, norms/retrieval/gates %.3fs\n",
                                  seconds[attention],seconds[experts],seconds[glue]);
    }
};
} // namespace mini_prefill

// Upper bound on the device scratch a prefill of `positions` tokens in chunks
// of `chunk` allocates: the planes below, moe_batch's expert-sorted rows and
// attn_batch's key window and score block. Zero once a prefill at least this
// large has run, because Engine scratch is retained between requests.
inline size_t inkling_prefill_scratch(const Engine& e, size_t positions, size_t chunk) {
    using mini_prefill::H;
    const size_t rows = std::min(positions,chunk), group = std::min<size_t>(rows,1024);
    if (!rows || (e.tmp_capacity("prefill.hidden") >= rows*H && e.tmp_capacity("attn_batch.keys") >= positions*1024))
        return 0;
    const size_t planes = 26*rows*H;
    const size_t experts = rows*8*(2*H + H/2) + size_t(H)*H;
    const size_t attention = rows*(3*3072 + 384 + 2304 + 1024 + 64) + 2*positions*1024
                           + 24*std::min<size_t>(size_t(8)<<20,group*positions) + group*24*1024;
    return (planes + experts + attention)*sizeof(float);
}

// A caller's `proceed` said to stop: its client is gone.
struct PrefillCancelled : std::runtime_error { PrefillCancelled() : std::runtime_error("prefill cancelled") {} };

// Evaluate positions [resume,ids.size()) of a prompt. resume is 0 for a fresh
// prompt, or a boundary whose state inkling_checkpoint::restore has put back.
// With `record`, saves the checkpoints a later prompt can resume from.
// `proceed` is asked after every layer visit whether to go on; deep in a long
// context one visit can take many seconds. The caller admits prompt + decode
// before this call. Continue with
// decode(e, returned_token, ids.size()).
inline int inkling_prefill(Engine& e, const std::vector<int>& ids, const float* image_device,
                           size_t rows, int chunk, int resume = 0, bool record = false,
                           const std::function<bool()>& proceed = {}) {
    using namespace mini_prefill;
    using mini_model::copy;
    if (ids.empty() || ids.size()>inkling_context::limit || chunk<1 || chunk>inkling_context::max_chunk)
        throw std::invalid_argument("invalid Inkling prefill length or chunk size");
    if (rows && !image_device) throw std::invalid_argument("missing projected image rows");
    if (resume<0 || size_t(resume)>=ids.size()) throw std::invalid_argument("Inkling prefill cannot resume there");
    std::vector<int> tokens(ids);
    size_t row=0;
    for (int& id : tokens) {
        if (id<0 || id>=201024) throw std::out_of_range("Inkling Mini token");
        if (id==200054 && row<rows) id=~int(row++);
    }
    if (row!=rows) throw std::invalid_argument("image placeholder count mismatch");
    const int length=int(ids.size()), capacity=std::min(length-resume,chunk);
    const auto checkpoints = record ? inkling_checkpoint::marks(resume,length) : std::vector<inkling_checkpoint::Mark>();
    auto tmp = [&](const char* name, size_t count) {
        return e.tmp(std::string("prefill.")+name,count);
    };
    auto planes = [&](const char* name, int count=1) { return tmp(name,count*capacity*H); };
    float *h=planes("hidden"), *pre=planes("pre"), *states=planes("states",5);
    float *prev=planes("prev",2), *msgs=planes("messages",4), *input=planes("input",3);
    float *qkv=planes("qkv",3), *residual=planes("residual"), *norm=planes("norm");
    float *concat=planes("concat",2), *low=tmp("low",capacity*rank);
    float *forget=planes("forget"), *gain=planes("gain");
    float *mlp=tmp("mlp",(capacity+3)*H);
    static_assert(sizeof(float)==sizeof(int));
    int* device_ids=reinterpret_cast<int*>(tmp("tokens",length));
    INKLING_HIP_CHECK(hipMemcpy(device_ids,tokens.data(),length*sizeof(int),hipMemcpyHostToDevice));
    const std::string root="model.llm.", ar=root+"attn_res.";
    int final_states=0, stride=0;
    Stages stages;
    for (int start=resume; start<length;) {
        int B=std::min(capacity,length-start);
        stride=B*H;
        std::vector<inkling_checkpoint::Mark> marks; // those this chunk passes
        for (const auto& mark : checkpoints)
            if (mark.boundary>start && mark.boundary<=start+B) marks.push_back(mark);
        embed_rows<<<B,T>>>(device_ids+start,image_device,e.w(root+"embed.weight"),
                            e.w(root+"embed_norm.weight"),h);
        auto retrieve = [&](const float* src, int n, const float* query, int count, float* dst) {
            retrieve_rows<<<dim3(B,count),T>>>(src,n,stride,query,dst);
        };
        auto gate = [&](const std::string& key, const float* x, const float* msg, float* y) {
            gate_input<<<dim3(B,2),T>>>(x,msg,concat);
            mini_dense::linear_bf16(e.w(key+"down_weight"),concat,low,B,2*H,rank);
            mini_dense::linear(e.w(key+"forget_weight"),low,forget,B,rank,H);
            mini_dense::linear(e.w(key+"gain_weight"),low,gain,B,rank,H);
            gate_mix<<<B*mini_model::blocks,T>>>(forget,gain,x,msg,y);
        };
        int n=0, visit=0;
        auto layer_step = [&](int layer, bool cross) {
            std::string idx=std::to_string(layer), key=root+"layers."+idx+".";
            bool full=layer%6==5, split=full || cross;
            norm_rows<<<B,T>>>(h,nullptr,states+n*stride,false);
            if (layer) retrieve(states,n+1,e.w(ar+"attn_queries."+std::to_string(layer-1)),
                                full ? 3 : 1,input);
            if (cross) {
                int off=(layer-6)*2*H;
                const float* qm=e.w(root+"middle_cycle_cross_pass_queries")+off;
                retrieve(prev,2,qm,1,msgs);
                retrieve(prev,2,e.w(root+"middle_cycle_cross_pass_kv_queries")+off,2,msgs+stride);
                retrieve(prev,2,qm+H,1,msgs+3*stride);
            }
            const char* controllers[]={"attn.","k.","v."};
            const char* norms[]={"q_input_norm.weight","k_input_norm.weight","v_input_norm.weight"};
            for (int i=0; i<(split ? 3 : 1); ++i) {
                const float* x=layer ? input+(full ? i*stride : 0) : h;
                if (cross) {
                    gate(key+"controllers."+controllers[i],x,msgs+i*stride,norm);
                    x=norm;
                }
                norm_rows<<<B,T>>>(x,e.w(key+(full ? norms[i] : "attn_norm.weight")),qkv+i*stride,true);
            }
            stages.lap(Stages::glue);
            attn_batch(e,layer,visit,start,qkv,split ? qkv+stride : qkv,
                       split ? qkv+2*stride : qkv,residual,B,marks);
            stages.lap(Stages::attention);
            if (layer%6==0) ++n;
            else mini_model::add<<<B*mini_model::blocks,T>>>(residual,h,residual);
            norm_rows<<<B,T>>>(residual,nullptr,states+n*stride,false);
            retrieve(states,n+1,e.w(ar+"mlp_queries."+idx),1,input);
            if (cross) gate(key+"controllers.mlp.",input,msgs+3*stride,input);
            norm_rows<<<B,T>>>(input,e.w(key+"mlp_norm.weight"),norm,true);
            stages.lap(Stages::glue);
            moe_batch(e,layer,norm,mlp+3*H,B);
            stages.lap(Stages::experts);
            // Prefix + raw MoE rows form one contiguous time series. Copying its
            // tail works even for B=1 or 2, and publishes scalar decode history.
            const std::string ring="model.mlp_history."+std::to_string(visit);
            float* hist=e.tmp(ring,3*H);
            if (start) copy(mlp,hist,3*H);
            else INKLING_HIP_CHECK(hipMemsetAsync(mlp,0,3*H*sizeof(float)));
            finish_rows<<<B*mini_model::blocks,T>>>(mlp,e.w(key+"mlp_sconv.weight"),residual,h);
            copy(hist,mlp+stride,3*H);
            for (const auto& mark : marks) // the three rows before the boundary
                copy(inkling_checkpoint::saved(e,ring,3*H,mark.slot),mlp+size_t(mark.boundary-start)*H,3*H);
            ++visit;
            // moe_batch read its routing back, so the device has finished this visit.
            if (proceed && !proceed()) throw PrefillCancelled();
        };
        for (int l=0; l<6; ++l) layer_step(l,false);
        copy(pre,h,stride);
        const int pre_blocks=n;
        for (int pass=0; pass<3; ++pass) {
            if (pass) {
                copy(prev,states+(pre_blocks+1)*stride,stride);
                norm_rows<<<B,T>>>(h,nullptr,prev+stride,false);
                n=pre_blocks;
                copy(h,pre,stride);
            }
            for (int l=6; l<18; ++l) layer_step(l,pass!=0);
        }
        for (int l=18; l<24; ++l) layer_step(l,false);
        final_states=n;
        start+=B;
    }
    const int last=stride-H;
    norm_rows<<<1,T>>>(h+last,nullptr,states+final_states*stride+last,false);
    retrieve_rows<<<1,T>>>(states+last,final_states+1,stride,e.w(ar+"output_query"),input);
    norm_rows<<<1,T>>>(input,e.w(root+"norm.weight"),norm,true);
    float* logits=e.tmp("model.logits",mini_model::vocab);
    e.mv(e.half(root+"unembed.weight"),norm,logits,mini_model::vocab,H);
    int* next=reinterpret_cast<int*>(e.tmp("model.id",1));
    mini_model::greedy<<<1,T>>>(logits,next);
    INKLING_HIP_CHECK(hipGetLastError());
    int id;
    INKLING_HIP_CHECK(hipMemcpy(&id,next,sizeof(id),hipMemcpyDeviceToHost));
    return id;
}
