// Host-only: the native chat rendering must reproduce the checkpoint's own
// chat template, token for token.
//   c++ -std=c++17 chat_template_test.cpp -lpcre2-8 -o chat-template-test
//   ./chat-template-test /path/to/Inkling-2-Mini ../../../../tools/inkling2_mini/chat-template-reference.json
// The reference comes from tools/inkling2_mini/make_chat_template_reference.py.
#include "tokenizer.hpp"
#include "serve_chat.hpp"
#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: chat-template-test CHECKPOINT REFERENCE.json");
    TextCodec codec(argv[1]);
    std::ifstream file(argv[2]);
    const auto reference = nlohmann::json::parse(file);
    int failures = 0;
    for (const auto& item : reference.at("cases")) {
        auto request = ink_http::parse_chat(item.at("request").dump(), "inkling");
        // The template writes one placeholder per image; the server widens it to the image's rows.
        for (auto& message : request.messages)
            for (auto& part : message.parts)
                if (part.image) part.rows = 1;
        const auto ids = codec.conversation(request.messages, request.effort, request.tools);
        const auto expected = item.at("token_ids").get<std::vector<int>>();
        size_t at = 0;
        while (at < ids.size() && at < expected.size() && ids[at] == expected[at]) ++at;
        const bool same = ids == expected;
        failures += !same;
        std::cout << (same ? "PASS " : "FAIL ") << item.at("name").get<std::string>() << " (" << expected.size() << " tokens)";
        if (!same) std::cout << ": first difference at " << at << ", got " << ids.size() << " tokens";
        std::cout << '\n';
    }
    return failures != 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
