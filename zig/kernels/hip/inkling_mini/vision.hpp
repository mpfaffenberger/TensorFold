#pragma once
#include "vision_ops.hpp"
#include "vision_png.hpp"
#include "dense.hpp"

// One encoded image per call. Returns contiguous host rows [ceil(H/40)*ceil(W/40),2304].
inline std::vector<float> inkling_project(Engine& e,const mini_vision::ImagePatches& image) {
    using namespace mini_vision;
    int tiles=image.rows*image.cols, n=4*tiles+1;
    auto tmp=[&](std::string key,int count){return e.tmp("vision."+key,count);};
    auto weight=[&](std::string key){return e.w("model.visual."+key);};
    auto linear=[&](std::string key,const float* x,float* y,int rows,int in,int out,bool bias,bool gelu=false){
        mini_dense::linear(weight(key+".weight"),x,y,rows,in,out);
        if(bias||gelu) activate<<<(rows*out+255)/256,256>>>(y,bias?weight(key+".bias"):nullptr,out,rows*out,gelu);
    };
    auto ln=[&](std::string key,const float* x,float* y){
        norm<<<n,256>>>(x,y,weight(key+".weight"),weight(key+".bias"));
    };
    float *pixels=tmp("pixels",image.values.size()),*x=tmp("hidden",n*1024),*z=tmp("branch",n*1024);
    INKLING_HIP_CHECK(hipMemcpy(pixels,image.values.data(),image.values.size()*sizeof(float),hipMemcpyHostToDevice));
    std::string emb="backbone.embeddings.";
    linear(emb+"patch_embedding",pixels,z+1024,n-1,1200,1024,false);
    position<<<dim3(n,4),256>>>(z,weight(emb+"position_embedding"),weight(emb+"class_embedding"),
                                 weight(emb+"cls_position_embedding"),image.rows,image.cols);
    ln("backbone.ln_pre",z,x);
    float *q=tmp("q",n*1024),*k=tmp("k",n*1024),*v=tmp("v",n*1024),*wide=tmp("wide",n*4096);
    for(int i=0;i<23;++i) {
        std::string layer="backbone.layers."+std::to_string(i)+".",att=layer+"self_attn.";
        ln(layer+"ln_1",x,z);
        linear(att+"q_proj",z,q,n,1024,1024,true);
        linear(att+"k_proj",z,k,n,1024,1024,true);
        linear(att+"v_proj",z,v,n,1024,1024,true);
        rope<<<dim3(n,2),256>>>(q,k,image.cols);
        mini_vision::attention<<<dim3(n,16),256,n*sizeof(float)>>>(q,k,v,z,n);
        linear(att+"out_proj",z,q,n,1024,1024,true);
        residual<<<n*4,256>>>(x,q,weight(layer+"ls_1.gamma"),n*1024);
        ln(layer+"ln_2",x,z);
        linear(layer+"mlp.c_fc",z,wide,n,1024,4096,true,true);
        linear(layer+"mlp.c_proj",wide,z,n,4096,1024,true);
        residual<<<n*4,256>>>(x,z,weight(layer+"ls_2.gamma"),n*1024);
    }
    // Skip CLS; the remaining consecutive four-row groups are already merger inputs.
    linear("proj_in",x+1024,wide,tiles,4096,4096,false,true);
    linear("proj_out",wide,z,tiles,4096,2304,false);
    INKLING_HIP_CHECK(hipGetLastError());
    std::vector<float> result(tiles*2304);
    INKLING_HIP_CHECK(hipMemcpy(result.data(),z,result.size()*sizeof(float),hipMemcpyDeviceToHost));
    return result;
}

inline std::vector<float> inkling_image(Engine& e,const std::vector<unsigned char>& encoded){return inkling_project(e,mini_vision::decode(encoded));}
