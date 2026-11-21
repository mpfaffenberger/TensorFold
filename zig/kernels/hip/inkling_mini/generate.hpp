#pragma once
#include "context.hpp"
#include "vision.hpp"
#include "model.hpp"
#include "prefill.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include "tokenizer.hpp"
// Runs one request on the serialized Engine. `ids` is the rendered
// conversation; projected rows of `image` replace (not add to) the embeddings
// of its placeholder tokens. on_token receives every generated token in order
// and returns false to stop early. Returns the number of tokens generated.
template<class OnToken>
inline int inkling_generate(Engine& e, const std::vector<int>& ids, const mini_vision::ImagePatches& image,
                            int max_tokens, OnToken on_token) {
    const size_t rows = size_t(image.rows)*image.cols;
    if (max_tokens < 1 || ids.empty()) throw std::invalid_argument("nothing to generate");
    // 0 selects the token-at-a-time reference prefill, the bit-exact oracle for the batched one.
    const char* requested = std::getenv("TF_INKLING_PREFILL_CHUNK");
    const int chunk = requested ? std::atoi(requested) : 4096;
    if (chunk < 0 || chunk > inkling_context::limit) throw std::invalid_argument("TF_INKLING_PREFILL_CHUNK must be 0..16384");
    inkling_admit(e, ids.size()+max_tokens, inkling_prefill_scratch(e, ids.size(), chunk));
    float* device = nullptr;
    if (rows) {
        auto projected = inkling_project(e, image);
        for (float& x : projected) { // BF16, like every other embedding row
            uint32_t bits; std::memcpy(&bits, &x, 4);
            bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
            std::memcpy(&x, &bits, 4);
        }
        device = e.tmp("image.rows", projected.size());
        INKLING_HIP_CHECK(hipMemcpy(device, projected.data(), projected.size()*sizeof(float), hipMemcpyHostToDevice));
    }
    const auto begin = std::chrono::steady_clock::now();
    int next = 0;
    if (chunk) next = inkling_prefill(e, ids, device, rows, chunk);
    else {
        size_t row = 0;
        for (size_t i = 0; i < ids.size(); ++i)
            next = mini_model::step(e, ids[i], int(i), ids[i] == 200054 && row < rows ? device+(row++)*mini_model::H : nullptr);
    }
    const auto first = std::chrono::steady_clock::now();
    if (const char* path = std::getenv("TF_INKLING_DUMP_LOGITS")) { // soft-capped next-token logits after prefill
        std::vector<float> logits(mini_model::vocab);
        INKLING_HIP_CHECK(hipMemcpy(logits.data(), e.tmp("model.logits", mini_model::vocab), logits.size()*sizeof(float), hipMemcpyDeviceToHost));
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(logits.data()), logits.size()*sizeof(float));
    }
    int position = int(ids.size()), generated = 0;
    for (;;) {
        ++generated;
        const bool wanted = on_token(next);
        if (next == 200006 || generated == max_tokens || !wanted) break;
        next = decode(e, next, position++);
    }
    if (std::getenv("TF_INKLING_TIMINGS")) {
        using seconds = std::chrono::duration<double>;
        std::fprintf(stderr, "Inkling %zu prompt tokens, prefill %.3fs; %d subsequent tokens, decode %.3fs\n", ids.size(),
                     seconds(first-begin).count(), generated-1, seconds(std::chrono::steady_clock::now()-first).count());
    }
    return generated;
}
