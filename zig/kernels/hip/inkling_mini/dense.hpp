#pragma once
#include "engine.hpp"
#include <hipblas/hipblas.h>

// Dense row-major linears through hipBLAS, for work that has many rows at
// once: batched prefill and the vision tower. The scalar decoder keeps
// Engine::mv, whose fixed reduction tree is the oracle the batched path is
// compared against; a single row gains nothing from a GEMM.
namespace mini_dense {
inline hipblasHandle_t blas() {
    // Leaked on purpose: a static destructor would race HIP runtime teardown.
    static hipblasHandle_t* handle = [] {
        auto* created = new hipblasHandle_t;
        if (hipblasCreate(created) != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("hipBLAS init failed");
        return created;
    }();
    return *handle;
}
// Column-major C[m,n] = op(A) op(B), on the default stream like every kernel here.
inline void gemm(hipblasOperation_t op_a, hipblasOperation_t op_b, int m, int n, int k,
                 const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    const float one = 1, zero = 0;
    if (m <= 0 || n <= 0 || k <= 0) return;
    if (hipblasSgemm(blas(),op_a,op_b,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc) != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipBLAS SGEMM failed");
}
// y[rows,out] = x[rows,in] * weight[out,in]^T; all three row-major and contiguous.
inline void linear(const float* weight, const float* x, float* y, int rows, int in, int out) {
    gemm(HIPBLAS_OP_T,HIPBLAS_OP_N,out,rows,in,weight,in,x,in,y,out);
}
static __global__ void round_bf16(float* y, size_t count) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned bits = __float_as_uint(y[i]);
    if ((bits & 0x7fffffff) > 0x7f800000) bits |= 0x00400000; // Engine::mv keeps NaNs
    else bits += 0x7fff + ((bits >> 16) & 1);                 // nearest, ties to even
    y[i] = __uint_as_float(bits & 0xffff0000);
}
// Engine::mv semantics for a whole batch: FP32 accumulate, BF16-rounded result.
inline void linear_bf16(const float* weight, const float* x, float* y, int rows, int in, int out) {
    linear(weight,x,y,rows,in,out);
    size_t count = size_t(rows)*out;
    round_bf16<<<unsigned((count+255)/256),256>>>(y,count);
}
} // namespace mini_dense
