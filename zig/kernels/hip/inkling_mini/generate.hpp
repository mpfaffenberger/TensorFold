#pragma once
#include "context.hpp"
#include "vision.hpp"
#include "model.hpp"
#include "prefill.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include "tokenizer.hpp"
// Projected visual rows replace (not add to) the normalized token embedding.
inline std::vector<int> inkling_generate(Engine& e,TextCodec& codec,const std::string& prompt,
    const std::vector<unsigned char>& png,int max_tokens,int context_limit) {
    mini_vision::ImagePatches patches{};
    if(!png.empty()) {
        try { patches=mini_vision::decode(png); }
        catch(const std::runtime_error& ex){ throw std::invalid_argument(ex.what()); }
    }
    size_t rows=patches.rows*patches.cols;
    auto ids=codec.turn(prompt,rows);
    if(max_tokens<1 || context_limit<1 || ids.size()+size_t(max_tokens)>size_t(context_limit))
        throw std::invalid_argument("prompt, image and output exceed the context budget");
    // 0 selects the token-at-a-time reference prefill, the bit-exact oracle for the batched one.
    const char* requested=std::getenv("TF_INKLING_PREFILL_CHUNK");
    const int chunk=requested ? std::atoi(requested) : 4096;
    if(chunk<0 || chunk>inkling_context::limit) throw std::invalid_argument("TF_INKLING_PREFILL_CHUNK must be 0..16384");
    inkling_admit(e,ids.size()+max_tokens,inkling_prefill_scratch(e,ids.size(),chunk));
    float* device=nullptr;
    if(rows){
        auto image=inkling_project(e,patches);
        for(float& x:image){uint32_t u;std::memcpy(&u,&x,4);u=(u+0x7fff+((u>>16)&1))&0xffff0000u;std::memcpy(&x,&u,4);}
        device=e.tmp("image.rows",image.size());
        INKLING_HIP_CHECK(hipMemcpy(device,image.data(),image.size()*sizeof(float),hipMemcpyHostToDevice));
    }
    const auto begin=std::chrono::steady_clock::now();
    int next=0;
    if(chunk) next=inkling_prefill(e,ids,device,rows,chunk);
    else {
        size_t row=0;
        for(size_t i=0;i<ids.size();++i)
            next=mini_model::step(e,ids[i],int(i),ids[i]==200054 && row<rows ? device+(row++)*mini_model::H : nullptr);
    }
    const auto first=std::chrono::steady_clock::now();
    if(const char* path=std::getenv("TF_INKLING_DUMP_LOGITS")) { // soft-capped next-token logits after prefill
        std::vector<float> logits(mini_model::vocab);
        INKLING_HIP_CHECK(hipMemcpy(logits.data(),e.tmp("model.logits",mini_model::vocab),logits.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::ofstream(path,std::ios::binary).write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));
    }
    int pos=int(ids.size());
    std::vector<int> out;
    while(true){out.push_back(next);if(next==200006 || int(out.size())==max_tokens)break;next=decode(e,next,pos++);}
    if(std::getenv("TF_INKLING_TIMINGS")) {
        using seconds=std::chrono::duration<double>;
        std::fprintf(stderr,"Inkling %zu prompt tokens, prefill %.3fs; %zu subsequent tokens, decode %.3fs\n",ids.size(),seconds(first-begin).count(),out.size()-1,seconds(std::chrono::steady_clock::now()-first).count());
    }
    return out;
}
