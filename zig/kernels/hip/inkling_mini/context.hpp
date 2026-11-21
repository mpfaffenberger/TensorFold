#pragma once
#include "engine.hpp"

namespace inkling_context {
// Implementation ceiling (tested cache length), not the checkpoint's 1M claim.
constexpr int limit = 16384, window = 1024, page = 64, kv = 1024, heads = 24;
// K/V are already BF16-rounded. Store their high halves losslessly, followed
// by float score scratch (reused in place by the sink softmax).
constexpr size_t page_floats = page * (kv + heads);
// Pages per visit. Sliding visits attend to `window` positions but keep every
// page too, so that a later request can resume at any earlier position.
constexpr int capacity = limit / page;
inline int first(bool full, int pos) { return full ? 0 : std::max(0, pos + 1 - window); }
inline float tau(bool full, int pos) {
    return full ? 1.f + .1f * std::log(std::max(1.f, (pos+1) / 8192.f)) : 1.f;
}
inline std::string state(int layer, int visit) {
    return "attn_state." + std::to_string(layer) + "." + std::to_string(visit) + ".";
}
// Physical layers: 0..5, then three passes over 6..17, then 18..23.
// Cache keys belong to visits, not just physical layers.
constexpr int visits = 48;
inline int layer_of(int visit) { return visit < 6 ? visit : visit < 42 ? 6 + (visit-6)%12 : visit-24; }
inline size_t missing_bytes(const Engine& e, size_t positions) {
    if (!positions || positions > limit) throw std::out_of_range("Inkling context requires 1..16384 positions");
    size_t bytes = e.missing_weight_bytes();
    for (int visit=0; visit<visits; ++visit) {
        const auto prefix = state(layer_of(visit),visit);
        const size_t pages = std::min<size_t>((positions+page-1)/page,capacity);
        for (size_t i=0; i<pages; ++i)
            if (e.tmp_capacity(prefix+"page."+std::to_string(i)) < page_floats) bytes += page_floats*sizeof(float);
    }
    return bytes;
}
inline void check_budget(size_t need, size_t avail) {
    constexpr size_t reserve = size_t(1)<<30;
    if (avail < reserve || need > avail-reserve)
        throw std::runtime_error("Inkling context: additional weight/cache bytes=" + std::to_string(need) +
                                 ", free=" + std::to_string(avail) + ", reserve=" + std::to_string(reserve));
}
}

// One serialized Engine. Call before each request with ALL positions including
// image patches and decode. Existing allocations already reduce free device RAM.
// Reserve 1 GiB for directories, small model/vision workspaces, HIP allocator
// granularity and other activity. Snapshot, not an allocation guarantee.
// scratch_bytes is working memory the request will allocate beyond weights and caches.
inline void inkling_admit(Engine& e, size_t positions, size_t scratch_bytes = 0) {
    size_t need = inkling_context::missing_bytes(e,positions) + scratch_bytes, avail, total;
    INKLING_HIP_CHECK(hipMemGetInfo(&avail,&total));
    inkling_context::check_budget(need,avail);
}
