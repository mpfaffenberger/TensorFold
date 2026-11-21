#pragma once
#include "attention_batch.hpp"
#include "moe_batch.hpp"
#include "prefill_ops.hpp"

// Start a fresh prompt at position zero. The caller admits prompt + decode
// before this call. Continue with decode(e, returned_token, ids.size()).
inline int inkling_prefill(Engine& e, const std::vector<int>& ids, const float* image_device,
                           size_t rows, int chunk) {
    using namespace mini_prefill;
    using mini_model::copy;
    if (ids.empty() || ids.size()>inkling_context::limit || chunk<1)
        throw std::invalid_argument("invalid Inkling prefill length or chunk size");
    if (rows && !image_device) throw std::invalid_argument("missing projected image rows");
    std::vector<int> tokens(ids);
    size_t row=0;
    for (int& id : tokens) {
        if (id<0 || id>=201024) throw std::out_of_range("Inkling Mini token");
        if (id==200054 && row<rows) id=~int(row++);
    }
    if (row!=rows) throw std::invalid_argument("image placeholder count mismatch");
    const int length=int(ids.size()), capacity=std::min(length,chunk);
    auto tmp = [&](const char* name, size_t count) {
        return e.tmp(std::string("prefill.")+name,count);
    };
    auto planes = [&](const char* name, int count=1) { return tmp(name,count*capacity*H); };
    float *h=planes("hidden"), *pre=planes("pre"), *states=planes("states",5);
    float *prev=planes("prev",2), *msgs=planes("messages",4), *input=planes("input",3);
    float *qkv=planes("qkv",3), *residual=planes("residual"), *norm=planes("norm");
    float *concat=planes("concat",2), *low=tmp("low",capacity*rank);
    float *mlp=tmp("mlp",(capacity+3)*H);
    static_assert(sizeof(float)==sizeof(int));
    int* device_ids=reinterpret_cast<int*>(tmp("tokens",length));
    INKLING_HIP_CHECK(hipMemcpy(device_ids,tokens.data(),length*sizeof(int),hipMemcpyHostToDevice));
    const std::string root="model.llm.", ar=root+"attn_res.";
    int final_states=0, stride=0;
    for (int start=0; start<length;) {
        int B=std::min(capacity,length-start);
        stride=B*H;
        embed_rows<<<B,T>>>(device_ids+start,image_device,e.w(root+"embed.weight"),
                            e.w(root+"embed_norm.weight"),h);
        auto retrieve = [&](const float* src, int n, const float* query, int count, float* dst) {
            retrieve_rows<<<dim3(B,count),T>>>(src,n,stride,query,dst);
        };
        auto gate = [&](const std::string& key, const float* x, const float* msg, float* y) {
            gate_input<<<dim3(B,2),T>>>(x,msg,concat);
            gate_down<<<dim3(rank,B),T>>>(e.w(key+"down_weight"),concat,low);
            gate_output<<<dim3(mini_model::blocks,B),T>>>(low,e.w(key+"forget_weight"),
                                                          e.w(key+"gain_weight"),x,msg,y);
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
            attn_batch(e,layer,visit,start,qkv,split ? qkv+stride : qkv,
                       split ? qkv+2*stride : qkv,residual,B);
            if (layer%6==0) ++n;
            else mini_model::add<<<B*mini_model::blocks,T>>>(residual,h,residual);
            norm_rows<<<B,T>>>(residual,nullptr,states+n*stride,false);
            retrieve(states,n+1,e.w(ar+"mlp_queries."+idx),1,input);
            if (cross) gate(key+"controllers.mlp.",input,msgs+3*stride,input);
            norm_rows<<<B,T>>>(input,e.w(key+"mlp_norm.weight"),norm,true);
            moe_batch(e,layer,norm,mlp+3*H,B);
            // Prefix + raw MoE rows form one contiguous time series. Copying its
            // tail works even for B=1 or 2, and publishes scalar decode history.
            float* hist=e.tmp("model.mlp_history."+std::to_string(visit),3*H);
            if (start) copy(mlp,hist,3*H);
            else INKLING_HIP_CHECK(hipMemsetAsync(mlp,0,3*H*sizeof(float)));
            finish_rows<<<B*mini_model::blocks,T>>>(mlp,e.w(key+"mlp_sconv.weight"),residual,h);
            copy(hist,mlp+stride,3*H);
            ++visit;
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
    e.mv(e.w(root+"unembed.weight"),norm,logits,mini_model::vocab,H);
    int* next=reinterpret_cast<int*>(e.tmp("model.id",1));
    mini_model::greedy<<<1,T>>>(logits,next);
    INKLING_HIP_CHECK(hipGetLastError());
    int id;
    INKLING_HIP_CHECK(hipMemcpy(&id,next,sizeof(id),hipMemcpyDeviceToHost));
    return id;
}
