#pragma once
#include "context.hpp"
#include <hip/hip_runtime.h>
#include <vector>

// Resuming a prompt partway. The state the model carries from one position to
// the next is the K/V pages, which every visit keeps for the whole context,
// and four small convolution rings per visit (raw K, V and attention-output
// rows of the last four positions, raw MoE rows of the last three). The rings
// are saved here as they stood after `boundary` positions: at every multiple
// of `spacing` and at the end of the prompt. A request that shares a prefix
// with the previous prompt restores the latest snapshot inside that prefix and
// evaluates only what follows.
namespace inkling_checkpoint {
constexpr int spacing = 256, hidden = 2304, kv = inkling_context::kv;
constexpr int slots = inkling_context::limit/spacing + 1; // one per grid boundary, one for a prompt end off the grid
struct Mark { int boundary, slot; };
inline int slot_of(int boundary) { return boundary%spacing ? slots-1 : boundary/spacing-1; }

// The snapshots a prefill of positions [resume,length) must write.
inline std::vector<Mark> marks(int resume, int length) {
    std::vector<Mark> list;
    for (int boundary=(resume/spacing+1)*spacing; boundary<length; boundary+=spacing) list.push_back({boundary,slot_of(boundary)});
    if (length>resume) list.push_back({length,slot_of(length)});
    return list;
}
// Where a new prompt may resume, given the prompt the snapshots belong to: the
// latest boundary inside their common prefix that leaves a token to evaluate.
inline int resume_point(const std::vector<int>& held, const std::vector<int>& prompt) {
    size_t common = 0;
    while (common<held.size() && common<prompt.size() && held[common]==prompt[common]) ++common;
    const size_t usable = std::min(common,prompt.empty() ? 0 : prompt.size()-1);
    if (!held.empty() && held.size()<=usable) return int(held.size());
    return int(usable/spacing)*spacing;
}

struct Ring { std::string name; int width; };
inline std::vector<Ring> rings(int visit) {
    const std::string state = inkling_context::state(inkling_context::layer_of(visit),visit);
    return {{state+"khist",4*kv},{state+"vhist",4*kv},{state+"ohist",4*hidden},
            {"model.mlp_history."+std::to_string(visit),3*hidden}};
}
// Where a ring's snapshot for `slot` lives.
inline float* saved(Engine& e, const std::string& ring, int width, int slot) {
    return e.tmp("checkpoint."+ring,size_t(slots)*width) + size_t(slot)*width;
}
// Device memory all snapshots take once allocated.
inline size_t bytes() {
    size_t floats = 0;
    for (const auto& ring : rings(0)) floats += size_t(slots)*ring.width;
    return floats*inkling_context::visits*sizeof(float);
}
// Put every ring back as it stood after `boundary` positions.
inline void restore(Engine& e, int boundary) {
    for (int visit=0; visit<inkling_context::visits; ++visit)
        for (const auto& ring : rings(visit))
            INKLING_HIP_CHECK(hipMemcpyAsync(e.tmp(ring.name,ring.width),saved(e,ring.name,ring.width,slot_of(boundary)),
                                             ring.width*sizeof(float),hipMemcpyDeviceToDevice));
}
} // namespace inkling_checkpoint
