#pragma once
#include "serve_http.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include <ctime>

namespace ink_http {
using Json = nlohmann::json;
inline std::vector<unsigned char> inline_image(std::string_view url) {
    size_t comma=url.find(',');
    if (comma==url.npos || url.substr(0,comma)!="data:image/png;base64")
        throw Error(400,"expected an inline base64 PNG");
    url.remove_prefix(comma+1);
    constexpr std::string_view digits="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (url.empty() || url.size()%4) throw Error(400,"invalid base64 length");
    std::vector<unsigned char> out; out.reserve(url.size()/4*3);
    for (size_t i=0;i<url.size();i+=4) {
        unsigned bits=0; int pad=0;
        for (int j=0;j<4;++j) {
            auto c=url[i+j]; bits<<=6;
            if (c=='=' && j>=2 && i+4==url.size()) { ++pad; continue; }
            auto d=digits.find(c);
            if (pad || d==digits.npos) throw Error(400,"invalid base64 image");
            bits|=unsigned(d);
        }
        if (pad && (bits & (pad==2?0xffff:0xff))) throw Error(400,"invalid base64 padding");
        out.push_back(bits>>16); if (pad<2) out.push_back(bits>>8); if (!pad) out.push_back(bits);
    }
    return out;
}
struct Turn { std::string prompt; std::vector<unsigned char> image; int tokens; };
inline Turn turn(const std::string& raw,const std::string& model,int cap) {
    size_t events=0;
    auto j=Json::parse(raw,[&](int depth, Json::parse_event_t, Json&) {
        if (depth>12 || ++events>256) throw Error(413,"chat JSON too complex");
        return true;
    });
    if (!j.is_object()) throw Error(400,"expected a chat object");
    for (auto it=j.begin();it!=j.end();++it) {
        const auto& k=it.key();
        if (k!="messages" && k!="model" && k!="stream" && k!="temperature" && k!="max_tokens" && k!="max_completion_tokens") throw Error(400,"unsupported chat option");
    }
    if (j.value("model",model)!=model) throw Error(404,"unknown model");
    if (j.value("stream",false) || j.value("temperature",0.0)!=0) throw Error(400,"only non-streaming greedy generation is supported");
    if (j.contains("max_tokens") && j.contains("max_completion_tokens")) throw Error(400,"specify one token limit");
    auto limit=j.value("max_tokens",j.value("max_completion_tokens",Json(std::min(cap,96))));
    if (!limit.is_number_integer() || limit<1 || limit>cap) throw Error(400,"max_tokens outside configured limit");
    Turn t{{},{},limit.get<int>()};
    const auto& msgs=j.at("messages");
    if (!msgs.is_array() || msgs.size()!=1 || msgs[0].at("role")!="user") throw Error(400,"expected one user message");
    const auto& content=msgs[0].at("content");
    if (content.is_string()) t.prompt=content.get<std::string>();
    else if (content.is_array()) {
        bool text=false;
        for (const auto& part:content) {
            if (part.at("type")=="text") { t.prompt+=part.at("text").get<std::string>(); text=true; }
            else if (part.at("type")=="image_url" && !text && t.image.empty())
                t.image=inline_image(part.at("image_url").at("url").get_ref<const std::string&>());
            else throw Error(400,"expected at most one inline image before text");
        }
    } else throw Error(400,"invalid content");
    if (t.prompt.empty() && t.image.empty()) throw Error(400,"empty user message");
    if (t.prompt.size()>32768) throw Error(413,"prompt exceeds 32768 bytes");
    return t;
}
template<class Codec> Json completion(const std::vector<int>& ids,Codec& codec,const std::string& model,int limit,unsigned long serial) {
    std::string text,thinking; std::string* channel=nullptr; bool stop=ids.size()<size_t(limit);
    for (int id:ids) {
        const auto& word=codec.word(id);
        if (word=="<|content_model_end_sampling|>" || word=="<|endoftext|>") { stop=true; break; }
        if (word=="<|content_thinking|>") channel=&thinking;
        else if (word=="<|content_text|>") channel=&text;
        else if (word=="<|end_message|>" || word=="<|message_model|>") channel=nullptr;
        else if (channel) *channel+=word;
    }
    Json msg={{"role","assistant"},{"content",text}};
    if (!thinking.empty()) msg["reasoning_content"]=thinking;
    Json choice={{"index",0},{"message",msg},{"finish_reason",stop?"stop":"length"}};
    return {{"id","chatcmpl-"+std::to_string(serial)},{"object","chat.completion"},{"created",std::time(nullptr)},
            {"model",model},{"choices",Json::array({choice})}};
}
} // namespace ink_http
