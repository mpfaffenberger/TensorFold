#pragma once
#include "chat.hpp"
#include "serve_http.hpp"
#include <nlohmann/json.hpp>
#include <ctime>

// OpenAI chat-completions requests and responses for the Inkling server.
// Generation is greedy: sampling parameters are accepted and ignored so that
// ordinary clients work. Anything that would change the answer's meaning and
// cannot be honoured (tools, stop sequences, JSON mode, n > 1) is refused.
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

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<unsigned char> image; // the one PNG its image part refers to
    std::string effort = "0";         // rendered into "Thinking effort level: ..."
    int max_tokens = 0;               // 0: the server's cap
    bool stream = false, usage_event = false;
};

// reasoning_effort as the chat template prints it: named levels or 0..0.99.
inline std::string thinking_effort(const Json& value) {
    if (value.is_string()) {
        const std::pair<const char*,const char*> levels[] = {{"none","0"},{"minimal","0.1"},{"low","0.2"},
            {"medium","0.7"},{"high","0.9"},{"xhigh","0.99"},{"max","0.99"}};
        for (const auto& [name,text] : levels) if (value==name) return text;
    } else if (value.is_number() && value.get<double>()>=0 && value.get<double>()<=0.99) {
        return value.get<double>()==0 ? "0" : Json(value.get<double>()).dump();
    }
    throw Error(400,"reasoning_effort must be none, minimal, low, medium, high, xhigh, max or 0..0.99");
}

inline ChatRequest parse_chat(const std::string& raw, const std::string& model) {
    constexpr size_t text_cap = 256*1024; // bounds tokenizer work; the context limit is enforced in tokens
    size_t events=0;
    auto j=Json::parse(raw,[&](int depth, Json::parse_event_t, Json&) {
        if (depth>16 || ++events>200000) throw Error(413,"chat JSON too complex");
        return true;
    });
    if (!j.is_object()) throw Error(400,"expected a chat object");
    auto absent = [&](const char* key) { return !j.contains(key) || j[key].is_null(); };
    if (j.value("model",model)!=model) throw Error(404,"unknown model");
    if (!absent("n") && j["n"]!=1) throw Error(400,"only n = 1 is supported");
    if (!absent("tools") && !j["tools"].empty()) throw Error(400,"tool calling is not supported by this server");
    if (!absent("stop") && !j["stop"].empty()) throw Error(400,"stop sequences are not supported by this server");
    if (!absent("response_format") && j["response_format"].value("type","text")!="text")
        throw Error(400,"only text responses are supported by this server");

    ChatRequest request;
    request.stream = !absent("stream") && j["stream"].get<bool>();
    if (!absent("stream_options")) request.usage_event = j["stream_options"].value("include_usage",false);
    if (!absent("reasoning_effort")) request.effort = thinking_effort(j["reasoning_effort"]);
    const char* limit = !absent("max_completion_tokens") ? "max_completion_tokens" : !absent("max_tokens") ? "max_tokens" : nullptr;
    if (limit) {
        if (!j[limit].is_number_integer() || j[limit]<1) throw Error(400,"max_tokens must be a positive integer");
        request.max_tokens = int(std::min<int64_t>(j[limit].get<int64_t>(),INT32_MAX));
    }

    if (absent("messages") || !j["messages"].is_array() || j["messages"].empty()) throw Error(400,"expected a non-empty messages array");
    size_t text_bytes=0;
    auto text_part = [&](const Json& text) {
        if (!text.is_string()) throw Error(400,"message text must be a string");
        if ((text_bytes += text.get_ref<const std::string&>().size()) > text_cap) throw Error(413,"messages exceed 262144 bytes of text");
        return ChatPart{false,text.get<std::string>(),0};
    };
    for (const auto& item : j["messages"]) {
        if (!item.is_object() || !item.contains("role") || !item["role"].is_string()) throw Error(400,"each message needs a role");
        ChatMessage message;
        message.role = item["role"].get<std::string>();
        if (message.role=="developer") message.role="system";
        if (message.role=="tool" || (item.contains("tool_calls") && !item["tool_calls"].is_null() && !item["tool_calls"].empty()))
            throw Error(400,"tool calling is not supported by this server");
        if (message.role!="system" && message.role!="user" && message.role!="assistant") throw Error(400,"unknown message role");
        if (message.role=="assistant" && item.contains("reasoning_content") && item["reasoning_content"].is_string())
            message.reasoning = text_part(item["reasoning_content"]).text;
        const Json content = item.contains("content") ? item["content"] : Json();
        if (content.is_string()) message.parts.push_back(text_part(content));
        else if (content.is_array()) {
            message.listed = true;
            for (const auto& part : content) {
                const std::string type = part.is_object() ? part.value("type","text") : "";
                if (part.is_string()) message.parts.push_back(text_part(part));
                else if (type=="text" || type=="input_text") message.parts.push_back(text_part(part.value("text",Json(""))));
                else if (type=="image_url" && message.role=="user" && request.image.empty() && part.contains("image_url") && part["image_url"].is_object() && part["image_url"].contains("url") && part["image_url"]["url"].is_string()) {
                    request.image = inline_image(part["image_url"]["url"].get_ref<const std::string&>());
                    message.parts.push_back(ChatPart{true,{},0});
                } else throw Error(400,"expected text parts and at most one inline PNG image in a user message");
            }
        } else if (!content.is_null()) throw Error(400,"message content must be a string, a part list or null");
        request.messages.push_back(std::move(message));
    }
    return request;
}

// Longest prefix that does not end inside a UTF-8 sequence. Tokens are byte
// strings, so a streamed character can arrive split across two tokens.
inline size_t complete_utf8(std::string_view bytes) {
    const size_t n = bytes.size();
    for (size_t back=1; back<=3 && back<=n; ++back) {
        const unsigned char c = bytes[n-back];
        if ((c & 0xC0)==0x80) continue; // continuation byte: its lead is further back
        const size_t need = c>=0xF0 ? 4 : c>=0xE0 ? 3 : c>=0xC0 ? 2 : 1;
        return need>back ? n-back : n;
    }
    return n;
}

// Routes generated tokens to the answer or the reasoning channel.
struct Splitter {
    enum Channel { none, answer, reasoning };
    Channel channel = none;
    bool finished = false; // the model ended its turn itself
    // Bytes this token adds to the current channel (empty for control tokens).
    template<class Codec> std::string_view feed(Codec& codec, int id) {
        const std::string& word = codec.word(id);
        if (word=="<|content_model_end_sampling|>" || word=="<|endoftext|>") { finished=true; channel=none; }
        else if (word=="<|content_thinking|>") channel=reasoning;
        else if (word=="<|content_text|>") channel=answer;
        else if (word=="<|end_message|>" || word=="<|message_model|>") channel=none;
        else if (channel!=none && !finished) return word;
        return {};
    }
};

inline Json usage(size_t prompt_tokens, size_t completion_tokens) {
    return {{"prompt_tokens",prompt_tokens},{"completion_tokens",completion_tokens},{"total_tokens",prompt_tokens+completion_tokens}};
}
inline Json envelope(const char* object, const std::string& model, unsigned long serial) {
    return {{"id","chatcmpl-"+std::to_string(serial)},{"object",object},{"created",std::time(nullptr)},{"model",model}};
}
inline std::string dump(const Json& value) { return value.dump(-1,' ',false,Json::error_handler_t::replace); }

template<class Codec> Json completion(const std::vector<int>& ids, Codec& codec, const std::string& model,
                                      size_t prompt_tokens, unsigned long serial) {
    std::string text, thinking;
    Splitter splitter;
    for (int id : ids) {
        auto bytes = splitter.feed(codec,id);
        (splitter.channel==Splitter::reasoning ? thinking : text).append(bytes);
    }
    Json message={{"role","assistant"},{"content",text}};
    if (!thinking.empty()) message["reasoning_content"]=thinking;
    Json body = envelope("chat.completion",model,serial);
    body["choices"] = Json::array({{{"index",0},{"message",message},{"finish_reason",splitter.finished?"stop":"length"}}});
    body["usage"] = usage(prompt_tokens,ids.size());
    return body;
}
inline Json chunk(const std::string& model, unsigned long serial, Json delta, const char* finish_reason = nullptr) {
    Json body = envelope("chat.completion.chunk",model,serial);
    body["choices"] = Json::array({{{"index",0},{"delta",std::move(delta)},{"finish_reason",finish_reason ? Json(finish_reason) : Json()}}});
    return body;
}
} // namespace ink_http
