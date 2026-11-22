// Host-only protocol checks: c++ -std=c++17 serve_test.cpp -o serve-test && ./serve-test
#include "serve_chat.hpp"
#include <cassert>
#include <iostream>
using namespace ink_http;

static Json user(Json content) { return {{"role","user"},{"content",std::move(content)}}; }
static std::string body(Json messages, Json extra = Json::object()) {
    extra["messages"] = std::move(messages);
    return extra.dump();
}
static int status_of(const std::string& raw) {
    try { parse_chat(raw,"test"); } catch (const Error& error) { return error.status; } catch (const Json::exception&) { return 400; }
    return 200;
}
// Stand-in vocabulary: control tokens, text tokens, a three-byte character split
// across two tokens, and the pieces of two tool calls.
struct Words {
    std::vector<std::string> table{"<|message_model|>","<|content_thinking|>","private","<|end_message|>","<|content_text|>",
                                   "answer","<|content_model_end_sampling|>","\xE2\x82","\xAC",
                                   "<|content_invoke_tool_json|>","get_","weather","{\"name\":\"get_weather\",\"args\":{\"city\":",
                                   "\"Paris\",\"unit\":\"c\"}}","{\"name\": oops","<|content_tool_error|>"};
    const std::string& word(int id) { return table.at(id); }
};

int main() {
    const Json image = {{"type","image_url"},{"image_url",{{"url","data:image/png;base64,AQI="}}}};
    const Json text = {{"type","text"},{"text","hello"}};

    // One image, then text, in a user message.
    auto request = parse_chat(body(Json::array({user(Json::array({image,text}))}),{{"max_tokens",12}}),"test");
    assert(request.image == std::vector<unsigned char>({1,2}) && request.max_tokens == 12 && !request.stream);
    assert(request.messages.size() == 1 && request.messages[0].listed && request.messages[0].parts.size() == 2);
    assert(request.messages[0].parts[0].image && request.messages[0].parts[1].text == "hello" && request.effort == "0");

    // A real conversation: system prompt, history with reasoning, sampling options a greedy server ignores.
    const Json history = Json::array({{{"role","developer"},{"content","Be brief."}}, user("Hi"),
        {{"role","assistant"},{"content","Hello."},{"reasoning_content","greet"}}, user("And now?")});
    request = parse_chat(body(history,{{"model","test"},{"stream",true},{"stream_options",{{"include_usage",true},{"continuous_usage_stats",true}}},
        {"temperature",0.7},{"top_p",0.9},{"seed",3},{"max_completion_tokens",99999},{"reasoning_effort","high"},{"tools",Json::array()},{"unknown_option",1}}),"test");
    assert(request.stream && request.usage_event && request.max_tokens == 99999 && request.effort == "0.9");
    assert(request.messages.size() == 4 && request.messages[0].role == "system" && request.messages[2].reasoning == "greet");
    assert(!request.messages[1].listed && request.messages[1].parts[0].text == "Hi");
    assert(thinking_effort(0.35) == "0.35" && thinking_effort(0) == "0" && thinking_effort("none") == "0");

    // Tools: declared with sorted keys, earlier calls rendered as the template prints them, results labelled.
    const Json tool = {{"type","function"},{"function",{{"name","get_weather"},{"parameters",{{"type","object"},{"properties",{{"city",{{"type","string"}}}}}}}}}};
    const Json called = {{"role","assistant"},{"content",nullptr},{"tool_calls",Json::array({
        {{"id","c1"},{"type","function"},{"function",{{"name","get_weather"},{"arguments","{\"unit\":\"c\",\"city\":\"Paris\",\"deep\":{\"z\":1,\"a\":2}}"}}}},
        {{"id","c2"},{"type","function"},{"function",{{"name","ping"},{"arguments",""}}}},
        {{"id","c3"},{"type","function"},{"function",{{"name","broken"},{"arguments","not json"}}}}})}};
    request = parse_chat(body(Json::array({user("Weather?"),called,{{"role","tool"},{"tool_call_id","c1"},{"content","21"}},
        {{"role","tool"},{"name","ping"},{"content",Json::array({text})}}}),{{"tools",Json::array({tool,{{"name","ping"}}})},{"tool_choice","auto"},{"parallel_tool_calls",true}}),"test");
    assert(request.tools == "[{\"description\":\"\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"type\":\"object\"},\"type\":\"function\"},"
                            "{\"description\":\"\",\"name\":\"ping\",\"parameters\":{},\"type\":\"function\"}]");
    assert(request.messages[1].calls.size() == 3 && request.messages[1].parts.empty());
    assert(request.messages[1].calls[0].body == "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"deep\":{\"z\":1,\"a\":2},\"unit\":\"c\"}}");
    assert(request.messages[1].calls[1].body == "{\"name\":\"ping\",\"args\":{}}");
    assert(request.messages[1].calls[2].body == "{\"name\":\"broken\",\"args\":{\"arguments\":\"not json\"}}");
    assert(request.messages[2].role == "tool" && request.messages[2].label == "get_weather" && request.messages[3].label == "ping");
    assert(parse_chat(body(Json::array({user("x")}),{{"tools",Json::array({tool})},{"tool_choice","none"}}),"test").tools.empty());
    assert(parse_chat(body(Json::array({user("x")}),{{"tools",Json::array()}}),"test").tools.empty());

    // Requests whose meaning cannot be honoured are refused, not silently altered.
    for (const auto& refused : {
            body(Json::array({user("x")}),{{"tools",Json::array({tool})},{"tool_choice","required"}}),
            body(Json::array({user("x")}),{{"tools",Json::array({tool})},{"tool_choice",{{"type","function"},{"function",{{"name","get_weather"}}}}}}),
            body(Json::array({user("x")}),{{"tools",Json::array({{{"type","function"},{"function",{{"description","nameless"}}}}})}}),
            body(Json::array({user("x"),{{"role","assistant"},{"tool_calls",Json::array({{{"id","c"},{"function",{{"arguments","{}"}}}}})}}})),
            body(Json::array({user("x")}),{{"stop",Json::array({"END"})}}),
            body(Json::array({user("x")}),{{"n",2}}),
            body(Json::array({user("x")}),{{"response_format",{{"type","json_object"}}}}),
            body(Json::array({user("x")}),{{"reasoning_effort","extreme"}}),
            body(Json::array({user("x")}),{{"max_tokens",0}}),
            body(Json::array({user(Json::array({image,image}))})),
            body(Json::array({{{"role","assistant"},{"content",Json::array({image})}}})),
            body(Json::array({{{"role","tool"},{"name","t"},{"content",Json::array({image})}}})),
            body(Json::array({user(Json::array({{{"type","image_url"},{"image_url",{{"url","https://example.com/a.png"}}}}}))})),
            body(Json::array()), std::string("[]")})
        assert(status_of(refused) == 400);
    assert(status_of(body(Json::array({user("x")}),{{"model","other"}})) == 404);
    assert(status_of(body(Json::array({user(std::string(5000000,'a'))}))) == 413);

    // Channels: reasoning and answer are separated, and the finish reason reflects who stopped.
    Words codec;
    auto done = completion({0,1,2,3,0,4,5,3,6},codec,"test",7,1);
    assert(done["choices"][0]["message"]["content"] == "answer" && done["choices"][0]["message"]["reasoning_content"] == "private");
    assert(done["choices"][0]["finish_reason"] == "stop" && done["usage"]["prompt_tokens"] == 7 && done["usage"]["completion_tokens"] == 9);
    assert(completion({0,4,5},codec,"test",7,2)["choices"][0]["finish_reason"] == "length");
    Splitter splitter;
    assert(splitter.feed(codec,0).empty() && splitter.feed(codec,4).empty() && splitter.channel == Splitter::answer);
    assert(splitter.feed(codec,5) == "answer" && splitter.feed(codec,6).empty() && splitter.finished);
    assert(splitter.feed(codec,5).empty() && std::string(splitter.finish_reason()) == "stop"); // nothing after the end

    // Tool calls: an answer, then a call spread over tokens, then a malformed one.
    done = completion({0,4,5,3, 0,10,11,9,12,13,3, 0,10,11,9,14,3, 6},codec,"test",7,3);
    const Json& message = done["choices"][0]["message"];
    assert(done["choices"][0]["finish_reason"] == "tool_calls" && message["content"] == "answer" && message["tool_calls"].size() == 2);
    assert(message["tool_calls"][0]["type"] == "function" && message["tool_calls"][0]["function"]["name"] == "get_weather");
    assert(message["tool_calls"][0]["function"]["arguments"] == "{\"city\":\"Paris\",\"unit\":\"c\"}");
    assert(message["tool_calls"][1]["function"]["name"] == "get_weather" && message["tool_calls"][1]["function"]["arguments"] == "{\"name\": oops");
    assert(message["tool_calls"][0]["id"] != message["tool_calls"][1]["id"] && message["tool_calls"][0]["id"].get<std::string>().rfind("call_",0) == 0);
    // A call with no answer has null content; a call cut off by the budget is still relayed, as "length".
    done = completion({0,10,11,9,12,13,3,6},codec,"test",7,4);
    assert(done["choices"][0]["message"]["content"].is_null() && done["choices"][0]["finish_reason"] == "tool_calls");
    done = completion({0,10,11,9,12},codec,"test",7,5);
    assert(done["choices"][0]["finish_reason"] == "length" && done["choices"][0]["message"]["tool_calls"].size() == 1);
    // A content kind this server does not relay is dropped, not mixed into the answer.
    done = completion({0,15,5,3,0,4,5,3,6},codec,"test",7,6);
    assert(done["choices"][0]["message"]["content"] == "answer" && !done["choices"][0]["message"].contains("tool_calls"));

    // A character split across tokens is held back until it is whole.
    assert(complete_utf8("ok") == 2 && complete_utf8("ok\xE2\x82") == 2 && complete_utf8("ok\xE2\x82\xAC") == 5);
    assert(complete_utf8("\xF0\x9F\x98") == 0 && complete_utf8("\xF0\x9F\x98\x80") == 4 && complete_utf8("") == 0);
    assert(chunk("test",1,{{"content","hi"}})["choices"][0]["finish_reason"].is_null());
    assert(chunk("test",1,Json::object(),"stop")["choices"][0]["finish_reason"] == "stop");
    std::cout << "serve protocol checks passed\n";
}
