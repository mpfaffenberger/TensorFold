// Host-only protocol checks: c++ -std=c++17 serve_test.cpp -o serve-test && ./serve-test
#include "serve_chat.hpp"
#include <cassert>
using namespace ink_http;
static std::string request(Json content,int limit=12) {
    return Json{{"max_tokens",limit},{"messages",Json::array({{{"role","user"},{"content",content}}})}}.dump();
}
int main() {
    Json image={{"type","image_url"},{"image_url",{{"url","data:image/png;base64,AQI="}}}};
    Json text={{"type","text"},{"text","hello"}};
    auto t=turn(request(Json::array({image,text})),"test",12);
    assert(t.image==std::vector<unsigned char>({1,2}) && t.prompt=="hello" && t.tokens==12);
    auto reject=[](const std::string& body){try { turn(body,"test",12); } catch (const Error&) { return; } assert(false);};
    reject(request(Json::array({text,image}))); reject(request(Json::array({image,image})));
    image["image_url"]["url"]="https://example.com/image.png";
    reject(request(Json::array({image,text}))); reject(request("hello",13));
    struct Words { std::string word(int i) { return std::vector<std::string>{"<|content_thinking|>","private","<|end_message|>","<|content_text|>","answer","<|content_model_end_sampling|>"}.at(i); } } codec;
    auto choice=completion({0,1,2,3,4,5},codec,"test",6,1)["choices"][0];
    assert(choice["message"]["content"]=="answer" && choice["message"]["reasoning_content"]=="private" && choice["finish_reason"]=="stop");
    assert(completion({0,1},codec,"test",2,2)["choices"][0]["finish_reason"]=="length");
}
