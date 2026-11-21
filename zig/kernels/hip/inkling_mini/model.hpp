#pragma once
// Sliding layers without messages share Q/K/V inputs. Full layers and later
// loop passes use the separate normalized inputs of attn_forward_qkv.
// visit is the chronological 0..47 execution index (unique cache per visit).
// attention returns *after* attn_sconv; moe returns *before* mlp_sconv.
#include "attention.hpp"
#include "moe.hpp"
#include "model_ops.hpp"

namespace mini_model {
inline void message_gate(Engine& e, const std::string& key, const float* x,
                         const float* msg, float* y) {
    float* concat = e.tmp("model.gate_input", 2*H);
    float* low = e.tmp("model.gate_low", rank);
    rms<<<1,T>>>(x, nullptr, concat, true);
    rms<<<1,T>>>(msg, nullptr, concat+H, true);
    e.mv(e.w(key+"down_weight"), concat, low, rank, 2*H);
    gate_up<<<blocks,T>>>(low, e.w(key+"forget_weight"), e.w(key+"gain_weight"), x, msg, y);
}

inline int step(Engine& e, int token, int pos, const float* image_row=nullptr) {
    if (token < 0 || token >= 201024 || pos < 0) throw std::out_of_range("Inkling Mini decode");
    const std::string root = "model.llm.";
    const std::string ar = root + "attn_res.";
    auto tmp = [&](const char* name, int n=H) { return e.tmp(std::string("model.")+name, n); };
    float* h = tmp("hidden"), *pre = tmp("pre_loop"), *states = tmp("states", 5*H);
    float* prev = tmp("previous", 2*H), *msgs = tmp("messages", 4*H);
    float* input = tmp("input", 3*H), *qkv = tmp("qkv", 3*H);
    float* residual = tmp("residual"), *mlp = tmp("mlp"), *norm = tmp("norm");
    if(image_row) copy(h,image_row,H);
    else rms<<<1,T>>>(e.w(root+"embed.weight") + size_t(token)*H, e.w(root+"embed_norm.weight"), h, true);

    int n = 0, visit = 0; // states[:n] holds normalized, completed blocks
    auto layer_step = [&](int layer, bool cross_pass) {
        const std::string idx = std::to_string(layer), key = root+"layers."+idx+".";
        const bool full = layer % 6 == 5;
        const int queries = full ? 3 : 1;
        rms<<<1,T>>>(h, nullptr, states+n*H, false);
        if (layer == 0) copy(input, h, H);
        else retrieve<<<queries,T>>>(states, n+1, e.w(ar+"attn_queries."+std::to_string(layer-1)), input);

        if (cross_pass) {
            const int offset = (layer-6)*2*H;
            const float* qm = e.w(root+"middle_cycle_cross_pass_queries") + offset;
            const float* kv = e.w(root+"middle_cycle_cross_pass_kv_queries") + offset;
            retrieve<<<1,T>>>(prev, 2, qm, msgs);
            retrieve<<<2,T>>>(prev, 2, kv, msgs+H);
            retrieve<<<1,T>>>(prev, 2, qm+H, msgs+3*H);
        }
        const char* controllers[] = {"attn.", "k.", "v."};
        const char* full_norms[] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
        const bool split = full || cross_pass;
        for (int i = 0; i < (split ? 3 : 1); ++i) {
            const float* x = full ? input+i*H : input;
            if (cross_pass) {
                message_gate(e, key+"controllers."+controllers[i], x, msgs+i*H, norm);
                x = norm;
            }
            rms<<<1,T>>>(x, e.w(key+(full ? full_norms[i] : "attn_norm.weight")), qkv+i*H, true);
        }
        if (split) attn_forward_qkv(e, layer, visit, pos, qkv, qkv+H, qkv+2*H, residual);
        else attn_forward_qkv(e, layer, visit, pos, qkv, qkv, qkv, residual);
        // Boundary: retain the incoming partial block and start a new one with
        // just the convolved attention output. Otherwise extend the partial.
        if (layer % 6 == 0) ++n;
        else add<<<blocks,T>>>(residual, h, residual);
        rms<<<1,T>>>(residual, nullptr, states+n*H, false);
        retrieve<<<1,T>>>(states, n+1, e.w(ar+"mlp_queries."+idx), input);
        if (cross_pass) message_gate(e, key+"controllers.mlp.", input, msgs+3*H, input);
        rms<<<1,T>>>(input, e.w(key+"mlp_norm.weight"), norm, true);
        moe(e, layer, norm, mlp);
        float* history = e.tmp("model.mlp_history."+std::to_string(visit), 3*H);
        finish_mlp<<<blocks,T>>>(mlp, e.w(key+"mlp_sconv.weight"), history, pos, residual, h);
        ++visit;
    };

    for (int l=0; l<6; ++l) layer_step(l, false);
    copy(pre, h, H);
    const int pre_blocks = n;
    for (int pass=0; pass<3; ++pass) {
        if (pass) {
            // HF slices previous_block_states[num_pre_loop_blocks+1:]:
            // exclude the duplicated loop entry, retain layer 12 entry + final h.
            copy(prev, states+(pre_blocks+1)*H, H);
            rms<<<1,T>>>(h, nullptr, prev+H, false);
            n = pre_blocks;
            copy(h, pre, H);
        }
        for (int l=6; l<18; ++l) layer_step(l, pass != 0);
    }
    for (int l=18; l<24; ++l) layer_step(l, false);
    rms<<<1,T>>>(h, nullptr, states+n*H, false);
    retrieve<<<1,T>>>(states, n+1, e.w(ar+"output_query"), input);
    rms<<<1,T>>>(input, e.w(root+"norm.weight"), norm, true);
    float* logits = tmp("logits", vocab);
    e.mv(e.half(root+"unembed.weight"), norm, logits, vocab, H);
    int* device_id = reinterpret_cast<int*>(tmp("id", 1));
    greedy<<<1,T>>>(logits, device_id);
    check(hipGetLastError());
    int id;
    check(hipMemcpy(&id, device_id, sizeof(id), hipMemcpyDeviceToHost));
    return id;
}
} // namespace mini_model

inline int decode(Engine& e, int token, int pos) { return mini_model::step(e, token, pos); }
