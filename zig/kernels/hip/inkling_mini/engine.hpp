#pragma once

#include "engine_checkpoint.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <memory>
#include <unordered_map>

namespace mini_engine {
inline void check(hipError_t error, const char* call) {
    if (error != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(error));
}
#define INKLING_HIP_CHECK(call) ::mini_engine::check((call), #call)

struct Free { void operator()(void* p) const noexcept { (void)hipFree(p); } };
using DevicePtr = std::unique_ptr<void, Free>;
inline DevicePtr allocate(size_t bytes) {
    void* p = nullptr;
    if (bytes) INKLING_HIP_CHECK(hipMalloc(&p, bytes));
    return DevicePtr(p);
}

// E4M3FN (not FNUZ): bias 7, signed zero, 0x7e = 448, 0x7f = NaN.
inline float e4m3(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 15 && m == 7 ? std::numeric_limits<float>::quiet_NaN() :
              std::ldexp(float(e ? 8+m : m), e ? e-10 : -9);
    return b & 128 ? -v : v;
}
inline void upload_floats(const Tensor& tensor, float* out) {
    if (tensor.type == Type::f32) {
        if (tensor.bytes) INKLING_HIP_CHECK(hipMemcpy(out,tensor.data,tensor.bytes,hipMemcpyHostToDevice));
        return;
    }
    // Bound staging memory independently of embedding / expert dimensions.
    std::vector<float> staging(std::min<size_t>(tensor.count, 1 << 18));
    for (size_t start = 0; start < tensor.count;) {
        size_t n = std::min(staging.size(), tensor.count-start);
        for (size_t i = 0; i < n; ++i) {
            size_t j = start+i;
            if (tensor.type == Type::bf16) {
                uint32_t bits = uint32_t(tensor.data[2*j]) << 16 | uint32_t(tensor.data[2*j+1]) << 24;
                std::memcpy(&staging[i], &bits, sizeof(bits));
            } else if (tensor.type == Type::u8) staging[i] = float(tensor.data[j]);
            else staging[i] = e4m3(tensor.data[j]);
        }
        INKLING_HIP_CHECK(hipMemcpy(out+start,staging.data(),n*sizeof(float),hipMemcpyHostToDevice));
        start += n;
    }
}

static __global__ void matvec(const float* weight, const float* x, float* y, int cols) {
    constexpr int lanes = 256;
    __shared__ float sums[lanes];
    const auto t = threadIdx.x;
    float sum = 0;
    for (size_t i = t; i < size_t(cols); i += lanes)
        sum = fmaf(weight[size_t(blockIdx.x)*cols+i], x[i], sum);
    sums[t] = sum;
    __syncthreads();
    for (int stride = lanes/2; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t+stride];
        __syncthreads();
    }
    if (!t) {
        unsigned bits = __float_as_uint(sums[0]);
        if ((bits & 0x7fffffff) > 0x7f800000) bits |= 0x00400000; // preserve NaNs
        else bits += 0x7fff + ((bits >> 16) & 1); // nearest, ties to even
        y[blockIdx.x] = __uint_as_float(bits & 0xffff0000);
    }
}
} // namespace mini_engine

class Engine {
    mini_engine::Checkpoint checkpoint;
    using DevicePtr = mini_engine::DevicePtr;
    struct Scratch { DevicePtr data; size_t capacity = 0; };
    std::unordered_map<std::string, DevicePtr> weights, packed;
    std::unordered_map<std::string, Scratch> scratch;
public:
    explicit Engine(std::string checkpoint_directory) : checkpoint(checkpoint_directory) {}
    ~Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    float* tmp(std::string name, size_t count) {
        auto& slot = scratch[name];
        if (count > slot.capacity) {
            auto bytes = mini_engine::product(count,sizeof(float));
            auto data = mini_engine::allocate(bytes);
            size_t old = slot.capacity*sizeof(float);
            if (old) INKLING_HIP_CHECK(hipMemcpy(data.get(),slot.data.get(),old,hipMemcpyDeviceToDevice));
            INKLING_HIP_CHECK(hipMemset(static_cast<char*>(data.get())+old,0,bytes-old));
            slot.data = std::move(data);
            slot.capacity = count;
        }
        return static_cast<float*>(slot.data.get());
    }
    const float* w(std::string full_tensor_key) {
        auto it = weights.find(full_tensor_key);
        if (it == weights.end()) {
            const auto& tensor = checkpoint.get(full_tensor_key);
            auto data = mini_engine::allocate(mini_engine::product(tensor.count,sizeof(float)));
            mini_engine::upload_floats(tensor,static_cast<float*>(data.get()));
            it = weights.emplace(std::move(full_tensor_key), std::move(data)).first;
        }
        return static_cast<const float*>(it->second.get());
    }
    const void* raw(std::string full_tensor_key) {
        auto it = packed.find(full_tensor_key);
        if (it == packed.end()) {
            const auto& tensor = checkpoint.get(full_tensor_key);
            auto data = mini_engine::allocate(tensor.bytes);
            // Never ask HIP to pin an entire file-backed expert matrix on an APU.
            const size_t chunk = 1 << 20;
            std::vector<uint8_t> stage(std::min(chunk, tensor.bytes));
            for (size_t offset = 0; offset < tensor.bytes;) {
                size_t len = std::min(chunk, tensor.bytes - offset);
                std::memcpy(stage.data(), tensor.data + offset, len);
                INKLING_HIP_CHECK(hipMemcpy(static_cast<uint8_t*>(data.get()) + offset,
                                            stage.data(), len, hipMemcpyHostToDevice));
                offset += len;
            }
            it = packed.emplace(std::move(full_tensor_key),std::move(data)).first;
        }
        return it->second.get();
    }
    void mv(const float* weight, const float* x, float* y, int rows, int cols) {
        mini_engine::require(rows >= 0 && cols >= 0, "negative matvec dimension");
        if (!rows) return;
        mini_engine::require(y && (!cols || (weight && x)), "null matvec operand");
        mini_engine::matvec<<<rows,256>>>(weight,x,y,cols);
        INKLING_HIP_CHECK(hipGetLastError());
    }
};
