// Parity of the batched prefill operators against the scalar decode kernels on
// identical inputs, on a real device with the real checkpoint:
//   hipcc -std=c++17 -O2 --offload-arch=gfx1151 batch_test.hip -lhipblas -o batch-test
//   ./batch-test /path/to/Inkling-2-Mini
// GEMMs reassociate sums, so results agree to BF16 rounding noise, not bit for bit.
#include "attention_batch.hpp"
#include "moe_batch.hpp"
#include <cstdio>
#include <vector>

constexpr int H = 2304;
static bool failed = false;

static float bf16(float value) {
    uint32_t bits; std::memcpy(&bits,&value,4);
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
    std::memcpy(&value,&bits,4); return value;
}
// Unit-RMS rows rounded to BF16, like every normalized operator input in the model.
static float* inputs(Engine& e, const std::string& name, int rows, uint32_t seed) {
    std::vector<float> host(size_t(rows)*H);
    for (int row = 0; row < rows; ++row) {
        double squares = 0;
        for (int i = 0; i < H; ++i) {
            seed = seed*1664525u + 1013904223u;
            float value = float(seed >> 8)/8388608.f - 1.f;
            host[size_t(row)*H+i] = value; squares += double(value)*value;
        }
        float scale = float(1./std::sqrt(squares/H));
        for (int i = 0; i < H; ++i) host[size_t(row)*H+i] = bf16(host[size_t(row)*H+i]*scale);
    }
    float* device = e.tmp(name,host.size());
    INKLING_HIP_CHECK(hipMemcpy(device,host.data(),host.size()*sizeof(float),hipMemcpyHostToDevice));
    return device;
}
static void compare(const char* what, const float* scalar, const float* batched, size_t count, double tolerance) {
    std::vector<float> a(count), b(count);
    INKLING_HIP_CHECK(hipMemcpy(a.data(),scalar,count*sizeof(float),hipMemcpyDeviceToHost));
    INKLING_HIP_CHECK(hipMemcpy(b.data(),batched,count*sizeof(float),hipMemcpyDeviceToHost));
    double worst = 0, error = 0, power = 0; size_t differing = 0;
    for (size_t i = 0; i < count; ++i) {
        double diff = double(a[i])-b[i];
        worst = std::max(worst,std::fabs(diff)); error += diff*diff; power += double(a[i])*a[i];
        differing += a[i] != b[i];
    }
    double relative = std::sqrt(error/std::max(power,1e-30));
    bool ok = relative <= tolerance && power > 0;
    failed |= !ok;
    std::printf("%-44s %s relative rms %.2e, max |diff| %.2e, output rms %.3f, %.2f%% elements differ\n",
                what, ok ? "PASS" : "FAIL", relative, worst, std::sqrt(power/count), 100.*differing/count);
}
// Scalar attention for every position under one visit key; batched under another, in two chunks.
static void attention_case(Engine& e, const char* what, int layer, int rows, int split) {
    float *q = inputs(e,"test.q",rows,11), *k = inputs(e,"test.k",rows,23), *v = inputs(e,"test.v",rows,37);
    float *scalar = e.tmp("test.scalar",size_t(rows)*H), *batched = e.tmp("test.batched",size_t(rows)*H);
    for (int pos = 0; pos < rows; ++pos)
        attn_forward_qkv(e,layer,46,pos,q+size_t(pos)*H,k+size_t(pos)*H,v+size_t(pos)*H,scalar+size_t(pos)*H);
    attn_batch(e,layer,47,0,q,k,v,batched,split);
    attn_batch(e,layer,47,split,q+size_t(split)*H,k+size_t(split)*H,v+size_t(split)*H,batched+size_t(split)*H,rows-split);
    compare(what,scalar,batched,size_t(rows)*H,2e-3);
    // The batched chunks must leave caches that scalar decode can continue from.
    float *next = inputs(e,"test.next",1,51), *after_scalar = e.tmp("test.after_scalar",H), *after_batched = e.tmp("test.after_batched",H);
    attn_forward_qkv(e,layer,46,rows,next,next,next,after_scalar);
    attn_forward_qkv(e,layer,47,rows,next,next,next,after_batched);
    compare("  scalar decode continuing from each cache",after_scalar,after_batched,H,2e-3);
}
static void moe_case(Engine& e, const char* what, int layer, int rows) {
    float* x = inputs(e,"test.q",rows,71);
    float *scalar = e.tmp("test.scalar",size_t(rows)*H), *batched = e.tmp("test.batched",size_t(rows)*H);
    for (int row = 0; row < rows; ++row) moe(e,layer,x+size_t(row)*H,scalar+size_t(row)*H);
    moe_batch(e,layer,x,batched,rows);
    compare(what,scalar,batched,size_t(rows)*H,2e-3);
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("usage: batch-test CHECKPOINT");
    Engine e(argv[1]);
    {   // Engine::mv against the GEMM wrapper it is replaced by.
        const std::string key = "model.llm.layers.0.attn.wq_du.weight";
        float *x = inputs(e,"test.q",3,5), *scalar = e.tmp("test.scalar",3*3072), *batched = e.tmp("test.batched",3*3072);
        for (int row = 0; row < 3; ++row) e.mv(e.w(key),x+row*H,scalar+row*3072,3072,H);
        mini_dense::linear_bf16(e.w(key),x,batched,3,H,3072);
        compare("dense projection, 3 rows",scalar,batched,3*3072,2e-3);
    }
    attention_case(e,"sliding attention, 40 rows",0,40,25);
    attention_case(e,"sliding attention across the 1024 window",0,1200,700);
    attention_case(e,"full attention, 300 rows",5,300,200);
    moe_case(e,"NVFP4 experts, direct rows (20)",0,20);
    moe_case(e,"NVFP4 experts, dequantize + GEMM (400)",0,400);
    moe_case(e,"MXFP8 experts, dequantize + GEMM (400)",20,400);
    return failed;
} catch (const std::exception& ex) { std::fprintf(stderr,"%s\n",ex.what()); return 2; }
