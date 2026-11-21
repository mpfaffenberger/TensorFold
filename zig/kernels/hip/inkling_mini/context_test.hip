// hipcc -std=c++17 -O2 --offload-arch=gfx1151 context_test.hip -o /tmp/context-test
// /tmp/context-test /path/to/Inkling-2-Mini
#include "attention.hpp"
#include <iostream>

static void check(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
static float bf(float f) {
    uint32_t u; std::memcpy(&u,&f,4);
    u = (u + 32767 + ((u>>16)&1)) & 0xffff0000u;
    std::memcpy(&f,&u,4); return f;
}
// One distinguished old key and a lag-zero bias, with group-specific values.
// Negative Vs and the sink make normalization, routing and packed loads observable.
static __global__ void token(int t, float* k, float* v) {
    int i = threadIdx.x + blockIdx.x*128;
    // Dense exactly representable products exercise all shuffle offsets.
    k[i] = t == 0 ? 1.f : ((i%8)-3.f)/32;
    v[i] = (i/128+1) * (t==0 ? -4.f : (t%3)-1.f);
}
static void exercise(Engine& e, bool full) {
    using namespace inkling_context;
    int cap = capacity;
    std::string prefix = state(full ? 5 : 0,full ? 5 : 0);
    auto tmp = [&](const char* k,int n){return e.tmp(k,n);};
    float *k=tmp("k",1024), *v=tmp("v",1024), *q=tmp("q",3072), *r=tmp("r",384), *tab=tmp("tab",16384);
    float *sink=tmp("sink",24), *gate=tmp("gate",3072), *y=tmp("y",3072);
    std::vector<float> ones(3072,1.f);
    float bias=3.f;
    INKLING_HIP_CHECK(hipMemcpy(tab,&bias,4,hipMemcpyHostToDevice));
    int end = full ? 16383 : 2112;
    for (int pos=0; pos<=end; ++pos) {
        token<<<8,128>>>(pos,k,v);
        auto pages=attn_cache_append(e,prefix,cap,pos,k,v);
        if (pos!=0 && pos!=1023 && pos!=1024 && pos!=1025 && pos!=8191 && pos!=8192 && pos!=end) continue;
        int lo=first(full,pos);
        float scale=tau(full,pos);
        check(std::abs(scale-(full ? 1+.1*std::log(std::max(1.,(pos+1)/8192.)) : 1)) < 1.e-7,"log scaling");
        INKLING_HIP_CHECK(hipMemcpy(q,ones.data(),3072*4,hipMemcpyHostToDevice));
        attn_norm<<<24,128>>>(q,nullptr,22.627417f,scale);
        INKLING_HIP_CHECK(hipMemcpy(r,ones.data(),384*4,hipMemcpyHostToDevice));
        attn_norm<<<24,16>>>(r,nullptr,8.f,scale);
        attn_scores<<<dim3((pos-lo+4)/4,24),128>>>(pages,cap,lo,pos,q,r,tab);
        attn_softmax<<<24,128>>>(pages,cap,lo,pos,sink);
        const int segments=(pos-lo)/attn_segment+1;
        float* partial=e.tmp("partial",size_t(segments)*3072);
        attn_mix<<<dim3(24,segments),128>>>(pages,cap,lo,pos,partial);
        attn_mix_finish<<<24,128>>>(partial,segments,gate,y);
        std::vector<float> host(3072);
        INKLING_HIP_CHECK(hipMemcpy(host.data(),y,host.size()*4,hipMemcpyDeviceToHost));
        // Independent chronological oracle, never wraps or reads device pages.
        std::vector<double> probs;
        double denom=1.; // sink=0
        for (int t=full ? 0 : std::max(0,pos-1023); t<=pos; ++t) {
            float dot=(t==0 ? 128 : 2)*bf(scale);
            probs.push_back(std::exp(double(bf(bf(dot/std::sqrt(128.f))+(t==pos?bf(3*bf(scale)):0)))));
            denom+=probs.back();
        }
        for (int g=0;g<8;++g) {
            // The kernel's order: keys in sequence within a segment, then segments in sequence.
            std::vector<float> parts(segments,0.f);
            for (size_t j=0;j<probs.size();++j) {
                int t=pos+1-int(probs.size())+int(j);
                parts[j/attn_segment] += bf(probs[j]/denom)*(g+1)*(t==0?-4.f:float(t%3)-1);
            }
            float sum=parts[0];
            for (int s=1;s<segments;++s) sum += parts[s];
            float expect=bf(bf(sum)*.5f);
            for (int i=g*384;i<(g+1)*384;++i) check(host[i]==expect,"GPU mix vs CPU oracle");
        }
        std::cout<<(full?"full":"ring")<<" pos="<<pos<<" output[0]="<<host[0]<<'\n';
    }
}
int main(int argc,char** argv) try {
    check(argc==2,"expected checkpoint directory"); Engine e(argv[1]);
    using namespace inkling_context;
    for (size_t n : {size_t(0),size_t(limit+1),SIZE_MAX}) {
        bool rejected=false; try {inkling_admit(e,n);} catch (const std::out_of_range&) {rejected=true;}
        check(rejected,"length rejection");
    }
    constexpr size_t reserve=size_t(1)<<30;
    check_budget(500,reserve+500);
    for (size_t free_bytes : {size_t(0),reserve-1,reserve+499}) {
        bool rejected=false; try {check_budget(500,free_bytes);} catch (const std::runtime_error&) {rejected=true;}
        check(rejected,"low-memory rejection");
    }
    inkling_admit(e,limit);
    size_t cold=missing_bytes(e,limit), weights=e.missing_weight_bytes();
    check(cold-weights==size_t(visits)*capacity*page_floats*4,"visit schedule"); // every visit keeps every page
    std::cout<<"cold weight bytes="<<weights<<" pages="<<cold-weights<<'\n';
    const char* image="model.visual.backbone.embeddings.class_embedding";
    e.raw(image); check(e.missing_weight_bytes()==weights,"wrong representation credited");
    e.w(image); check(e.missing_weight_bytes()==weights-1024*4,"image weight not credited");
    exercise(e,false); exercise(e,true);
    // The sliding exercise reaches position 2112 (34 pages); the full one fills all 256.
    check(missing_bytes(e,limit)==cold-4096-(34+256)*page_floats*4,"warm cache accounting");
    check(missing_bytes(e,1)==weights-4096+46*page_floats*4,"shorter request accounting");
    exercise(e,false); // restart at zero after a longer request without clearing pages
    inkling_admit(e,1); std::cout<<"PASS\n";
} catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
