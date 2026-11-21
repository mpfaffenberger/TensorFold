// Standalone checkpoint + GPU primitive smoke test; see engine_build.md.
#include "engine.hpp"
#include <array>
#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("usage: engine-smoke CHECKPOINT_DIR");
    Engine e(argv[1]);
    auto assert_ok = [](bool ok) { if (!ok) throw std::runtime_error("engine smoke mismatch"); };
    auto* a = e.tmp("matrix", 6);
    for (int i = 0; i < 128; ++i) e.tmp(std::to_string(i), 1);
    assert_ok(e.tmp("matrix", 6) == a);
    float host[6];
    INKLING_HIP_CHECK(hipMemcpy(host,a,sizeof(host),hipMemcpyDeviceToHost));
    for (auto v : host) assert_ok(v == 0);
    const float matrix[] = {1, 0.00390625f, 0, -1, -0.01171875f, 0};
    const float vector[] = {1, 1, 1};
    auto* x = e.tmp("input", 3);
    auto* y = e.tmp("output", 2);
    INKLING_HIP_CHECK(hipMemcpy(a,matrix,sizeof(matrix),hipMemcpyHostToDevice));
    INKLING_HIP_CHECK(hipMemcpy(x,vector,sizeof(vector),hipMemcpyHostToDevice));
    e.mv(a,x,y,2,3); // both rows are ties; one rounds up in magnitude, one down
    INKLING_HIP_CHECK(hipMemcpy(host,y,2*sizeof(float),hipMemcpyDeviceToHost));
    assert_ok(host[0] == 1.f && host[1] == -1.015625f);

    for (const auto& [key, width] : std::array<std::pair<const char*, int>,4>{{
        {"model.llm.layers.0.attn.rel_l2_clip.scale",2},
        {"model.llm.attn_res.attn_queries.0",4},
        {"model.llm.layers.20.mlp.experts.w2_weight_scale",1},
        {"model.llm.layers.0.mlp.experts.w2_weight_scale",1}}}) {
        const void* raw = e.raw(key);
        auto* w = e.w(key);
        assert_ok(raw == e.raw(key) && w == e.w(key));
        uint8_t bytes[4]{};
        INKLING_HIP_CHECK(hipMemcpy(bytes,raw,width,hipMemcpyDeviceToHost));
        INKLING_HIP_CHECK(hipMemcpy(host,w,sizeof(float),hipMemcpyDeviceToHost));
        float expected;
        if (width == 4) std::memcpy(&expected,bytes,4);
        else if (width == 2) {
            uint32_t bits = (uint32_t(bytes[1]) << 24) | (uint32_t(bytes[0]) << 16);
            std::memcpy(&expected,&bits,4);
        } else if (std::string(key).find("layers.20.") != std::string::npos) expected = bytes[0];
        else expected = mini_engine::e4m3(bytes[0]);
        assert_ok(expected == host[0]);
    }
    std::cout << "Engine smoke passed\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
