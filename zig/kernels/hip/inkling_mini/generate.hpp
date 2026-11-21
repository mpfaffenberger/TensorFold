#pragma once
#include "context.hpp"
#include "vision.hpp"
#include "model.hpp"
#include "prefill.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include "tokenizer.hpp"
// The prompt whose state the Engine still holds from the last request. A
// request that starts the same way resumes after the shared part, which is what
// makes an agent loop workable: its system prompt and tool list never change.
struct PrefixCache { std::vector<int> prompt; };

// Runs one request on the serialized Engine. `ids` is the rendered
// conversation; projected rows of `image` replace (not add to) the embeddings
// of its placeholder tokens. on_token receives every generated token in order
// and returns false to stop early. Returns the number of tokens generated.
template<class OnToken>
inline int inkling_generate(Engine& e, const std::vector<int>& ids, const mini_vision::ImagePatches& image,
                            int max_tokens, OnToken on_token, PrefixCache* cache = nullptr) {
    const size_t rows = size_t(image.rows)*image.cols;
    if (max_tokens < 1 || ids.empty()) throw std::invalid_argument("nothing to generate");
    // 0 selects the token-at-a-time reference prefill, the bit-exact oracle for the batched one.
    const char* requested = std::getenv("TF_INKLING_PREFILL_CHUNK");
    const int chunk = requested ? std::atoi(requested) : 4096;
    if (chunk < 0 || chunk > inkling_context::limit) throw std::invalid_argument("TF_INKLING_PREFILL_CHUNK must be 0..16384");
    // Image rows are not part of the token sequence, so prompts with one are neither resumed nor remembered.
    const char* reuse_setting = std::getenv("TF_INKLING_PREFIX_CACHE");
    const bool reuse = cache && chunk && !rows && !(reuse_setting && std::string(reuse_setting) == "0");
    const int resume = reuse ? inkling_checkpoint::resume_point(cache->prompt, ids) : 0;
    if (cache) cache->prompt.clear(); // this request overwrites the state it described
    size_t scratch = inkling_prefill_scratch(e, ids.size()-resume, chunk);
    if (reuse && !e.tmp_capacity("checkpoint.model.mlp_history.0")) scratch += inkling_checkpoint::bytes();
    inkling_admit(e, ids.size()+max_tokens, scratch);
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
    if (chunk) {
        if (resume) inkling_checkpoint::restore(e, resume);
        next = inkling_prefill(e, ids, device, rows, chunk, resume, reuse);
        if (reuse) cache->prompt = ids;
    }
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
        std::fprintf(stderr, "Inkling %zu prompt tokens (%d reused), prefill %.3fs; %d subsequent tokens, decode %.3fs\n", ids.size(),
                     resume, seconds(first-begin).count(), generated-1, seconds(std::chrono::steady_clock::now()-first).count());
    }
    return generated;
}
