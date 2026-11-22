#pragma once
#include "chat.hpp"
#include "serve_http.hpp"
#include <nlohmann/json.hpp>
#include <ctime>
#include <map>
#include <random>

// OpenAI chat-completions requests and responses for the Inkling server.
// Generation is greedy: sampling parameters are accepted and ignored so that
// ordinary clients work. Anything that would change the answer's meaning and
// cannot be honoured (a forced tool choice, stop sequences, JSON mode, n > 1)
// is refused.
namespace ink_http {
using Json = nlohmann::json;            // responses; keys print sorted
using Ordered = nlohmann::ordered_json; // requests: the template prints tool arguments in the order given

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
    std::string tools;                // tool declaration JSON, empty for none
    std::string effort = "0";         // rendered into "Thinking effort level: ..."
    int max_tokens = 0;               // 0: the server's cap
    bool stream = false, usage_event = false;
};

// reasoning_effort as the chat template prints it: named levels or 0..0.99.
inline std::string thinking_effort(const Ordered& value) {
    if (value.is_string()) {
        const std::pair<const char*,const char*> levels[] = {{"none","0"},{"minimal","0.1"},{"low","0.2"},
            {"medium","0.7"},{"high","0.9"},{"xhigh","0.99"},{"max","0.99"}};
        for (const auto& [name,text] : levels) if (value==name) return text;
    } else if (value.is_number() && value.get<double>()>=0 && value.get<double>()<=0.99) {
        return value.get<double>()==0 ? "0" : Json(value.get<double>()).dump();
    }
    throw Error(400,"reasoning_effort must be none, minimal, low, medium, high, xhigh, max or 0..0.99");
}

inline std::string string_field(const Ordered& object, const char* key) {
    return object.is_object() && object.contains(key) && object[key].is_string() ? object[key].get<std::string>() : std::string();
}

// The template's tool declaration: one {description, name, parameters, type}
// per tool, every key sorted, no whitespace.
inline std::string declare_tools(const Ordered& tools) {
    Json specs = Json::array();
    for (const auto& tool : tools) {
        const Ordered& function = tool.is_object() && tool.contains("function") ? tool["function"] : tool;
        const std::string name = string_field(function,"name");
        if (name.empty()) throw Error(400,"each tool needs a function name");
        const std::string type = string_field(tool,"type");
        Json parameters = Json::object();
        if (function.contains("parameters") && function["parameters"].is_object()) parameters = Json(function["parameters"]);
        specs.push_back({{"description",string_field(function,"description")},{"name",name},
                         {"parameters",std::move(parameters)},{"type",type.empty() ? "function" : type}});
    }
    return specs.dump();
}

// The template's body for one earlier tool call: top-level argument names
// sorted, values as given. Clients send the arguments as a JSON string. A
// string that is not a JSON object (a call the model got wrong earlier) is
// kept under "arguments" so the conversation can continue.
inline std::string invoke_body(const std::string& name, const Ordered& given) {
    Ordered arguments = Ordered::object();
    if (given.is_object()) arguments = given;
    else if (given.is_string()) {
        const std::string text = given.get<std::string>();
        Ordered parsed = text.empty() ? Ordered::object() : Ordered::parse(text,nullptr,false);
        if (parsed.is_object()) arguments = std::move(parsed);
        else arguments["arguments"] = text;
    } else if (!given.is_null()) throw Error(400,"tool call arguments must be an object or a JSON string");
    std::map<std::string,const Ordered*> sorted;
    for (auto item=arguments.begin(); item!=arguments.end(); ++item) sorted.emplace(item.key(),&item.value());
    std::string body = "{\"name\":" + Json(name).dump() + ",\"args\":{";
    for (const auto& [key,value] : sorted) {
        if (body.back()!='{') body += ',';
        body += Json(key).dump() + ":" + value->dump();
    }
    return body + "}}";
}

inline ChatRequest parse_chat(const std::string& raw, const std::string& model) {
    constexpr size_t text_cap = 4*1024*1024; // bounds tokenizer work; the context limit is enforced in tokens
    size_t events=0;
    auto j=Ordered::parse(raw,[&](int depth, Ordered::parse_event_t, Ordered&) {
        if (depth>32 || ++events>400000) throw Error(413,"chat JSON too complex");
        return true;
    });
    if (!j.is_object()) throw Error(400,"expected a chat object");
    auto absent = [&](const char* key) { return !j.contains(key) || j[key].is_null(); };
    if (j.value("model",model)!=model) throw Error(404,"unknown model");
    if (!absent("n") && j["n"]!=1) throw Error(400,"only n = 1 is supported");
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
    // Nothing here can force a call, so only the choices that leave it to the model are accepted.
    bool declare = !absent("tools") && j["tools"].is_array() && !j["tools"].empty();
    if (!absent("tool_choice")) {
        if (j["tool_choice"]=="none") declare = false;
        else if (j["tool_choice"]!="auto") throw Error(400,"tool_choice must be auto or none: this server cannot force a tool call");
    }
    if (declare) request.tools = declare_tools(j["tools"]);

    if (absent("messages") || !j["messages"].is_array() || j["messages"].empty()) throw Error(400,"expected a non-empty messages array");
    size_t text_bytes=0;
    auto text_part = [&](const Ordered& text) {
        if (!text.is_string()) throw Error(400,"message text must be a string");
        if ((text_bytes += text.get_ref<const std::string&>().size()) > text_cap) throw Error(413,"messages exceed 4 MiB of text");
        return ChatPart{false,text.get<std::string>(),0};
    };
    for (const auto& item : j["messages"]) {
        if (!item.is_object() || !item.contains("role") || !item["role"].is_string()) throw Error(400,"each message needs a role");
        ChatMessage message;
        message.role = item["role"].get<std::string>();
        if (message.role=="developer") message.role="system";
        if (message.role!="system" && message.role!="user" && message.role!="assistant" && message.role!="tool") throw Error(400,"unknown message role");
        if (message.role=="assistant") {
            message.label = string_field(item,"name");
            if (item.contains("reasoning_content") && item["reasoning_content"].is_string())
                message.reasoning = text_part(item["reasoning_content"]).text;
            if (item.contains("tool_calls") && item["tool_calls"].is_array())
                for (const auto& call : item["tool_calls"]) {
                    const Ordered function = call.is_object() && call.contains("function") ? call["function"] : Ordered();
                    const std::string name = string_field(function,"name");
                    if (name.empty()) throw Error(400,"each tool call needs a function name");
                    message.calls.push_back({string_field(call,"id"),name,
                        invoke_body(name,function.contains("arguments") ? function["arguments"] : Ordered())});
                    text_bytes += message.calls.back().body.size();
                }
        } else if (message.role=="tool") {
            // A result names its tool directly or through the call it answers.
            message.label = string_field(item,"name");
            const std::string answered = string_field(item,"tool_call_id");
            if (message.label.empty() && !answered.empty())
                for (const auto& earlier : request.messages)
                    for (const auto& call : earlier.calls)
                        if (call.id==answered) message.label += call.name;
        }
        const Ordered content = item.contains("content") ? item["content"] : Ordered();
        if (content.is_string()) message.parts.push_back(text_part(content));
        else if (content.is_array()) {
            message.listed = true;
            for (const auto& part : content) {
                const std::string type = part.is_object() ? part.value("type","text") : "";
                if (part.is_string()) message.parts.push_back(text_part(part));
                else if (type=="text" || type=="input_text") message.parts.push_back(text_part(part.contains("text") ? part["text"] : Ordered("")));
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

// A tool call the model made, in OpenAI's form.
struct MadeCall { std::string id, name, arguments; };

// The model prints a call as {"name":...,"args":{...}} after a header naming
// the tool. Malformed JSON is passed on as the arguments so the client's own
// validation can ask the model to try again.
inline MadeCall made_call(const std::string& header, const std::string& body) {
    static std::mt19937_64 random{std::random_device{}()};
    char id[32];
    std::snprintf(id,sizeof(id),"call_%016llx",static_cast<unsigned long long>(random()));
    const Ordered parsed = Ordered::parse(body,nullptr,false);
    if (parsed.is_object() && parsed.contains("args") && parsed["args"].is_object()) {
        const std::string name = string_field(parsed,"name");
        return {id,name.empty() ? header : name,parsed["args"].dump()};
    }
    return {id,header,body};
}

// Routes generated tokens to the answer, the reasoning or a tool call.
struct Splitter {
    enum Channel { none, answer, reasoning, tool };
    Channel channel = none;
    bool finished = false; // the model ended its turn itself
    std::vector<MadeCall> calls;
    // Bytes this token adds to the answer or the reasoning (empty otherwise).
    template<class Codec> std::string_view feed(Codec& codec, int id) {
        const std::string& word = codec.word(id);
        if (finished) return {};
        const bool control = word.size()>4 && word.compare(0,2,"<|")==0 && word.compare(word.size()-2,2,"|>")==0;
        if (word=="<|content_model_end_sampling|>" || word=="<|endoftext|>") { close(); finished=true; }
        else if (word=="<|message_model|>") { close(); in_header=true; header.clear(); }
        else if (word=="<|content_thinking|>") open(reasoning);
        else if (word=="<|content_text|>") open(answer);
        else if (word=="<|content_invoke_tool_json|>") open(tool);
        else if (control) close(); // end of message, or a kind this server does not relay
        else if (in_header) header += word;
        else if (channel==tool) body += word;
        else if (channel!=none) return word;
        return {};
    }
    // Call once generation has stopped: a call cut off by the token budget still counts.
    void finish() { close(); }
    const char* finish_reason() const { return !finished ? "length" : calls.empty() ? "stop" : "tool_calls"; }
private:
    bool in_header = false;
    std::string header, body; // text after <|message_model|>, and the call being printed
    void open(Channel next) { close(); channel=next; }
    void close() {
        if (channel==tool) calls.push_back(made_call(header,body));
        channel=none; in_header=false; body.clear();
    }
};

inline Json usage(size_t prompt_tokens, size_t completion_tokens) {
    return {{"prompt_tokens",prompt_tokens},{"completion_tokens",completion_tokens},{"total_tokens",prompt_tokens+completion_tokens}};
}
inline Json envelope(const char* object, const std::string& model, unsigned long serial) {
    return {{"id","chatcmpl-"+std::to_string(serial)},{"object",object},{"created",std::time(nullptr)},{"model",model}};
}
inline Json tool_call(const MadeCall& call) {
    return {{"id",call.id},{"type","function"},{"function",{{"name",call.name},{"arguments",call.arguments}}}};
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
    splitter.finish();
    Json message={{"role","assistant"},{"content",text.empty() && !splitter.calls.empty() ? Json() : Json(text)}};
    if (!thinking.empty()) message["reasoning_content"]=thinking;
    for (const auto& call : splitter.calls) message["tool_calls"].push_back(tool_call(call));
    Json body = envelope("chat.completion",model,serial);
    body["choices"] = Json::array({{{"index",0},{"message",message},{"finish_reason",splitter.finish_reason()}}});
    body["usage"] = usage(prompt_tokens,ids.size());
    return body;
}
inline Json chunk(const std::string& model, unsigned long serial, Json delta, const char* finish_reason = nullptr) {
    Json body = envelope("chat.completion.chunk",model,serial);
    body["choices"] = Json::array({{{"index",0},{"delta",std::move(delta)},{"finish_reason",finish_reason ? Json(finish_reason) : Json()}}});
    return body;
}
} // namespace ink_http
