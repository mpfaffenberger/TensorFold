#pragma once
// OpenAI-style chat server for one resident Engine. Included by main.hip after
// generate.hpp. Requests are served one at a time, in arrival order.
#include "serve_chat.hpp"
#include <cstdlib>
#include <iostream>

namespace ink_http {
struct ServeOptions {
    std::string checkpoint, host = "127.0.0.1", model = "Inkling-2-Mini", key;
    int port = 8080, context = 4096, max_tokens = 4096;
    bool vision = false;
    bool loopback() const { return host == "127.0.0.1" || host == "localhost"; }
};

inline ServeOptions serve_options(int argc, char** argv) {
    if (argc < 3)
        throw std::invalid_argument("usage: tf-inkling-hip serve CHECKPOINT [--host ADDRESS] [--port N] "
                                    "[--context N] [--max-tokens N] [--name MODEL] [--vision]");
    ServeOptions options;
    options.checkpoint = argv[2];
    if (const char* key = std::getenv("TENSORFOLD_API_KEY")) options.key = key;
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--vision") { options.vision = true; continue; }
        if (i+1 >= argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--host") options.host = value;
        else if (flag == "--port") options.port = int(number(value,65535));
        else if (flag == "--context") options.context = int(number(value,inkling_context::limit));
        else if (flag == "--max-tokens") options.max_tokens = int(number(value,inkling_context::limit));
        else if (flag == "--name") options.model = value;
        else throw std::invalid_argument("unsupported serve flag: " + flag);
    }
    if (!options.port || !options.context || !options.max_tokens || options.model.empty())
        throw std::invalid_argument("port, context, token cap and model name must be nonempty/nonzero");
    // Nothing else stands between the network and the GPU.
    if (!options.loopback() && options.key.empty())
        throw std::invalid_argument("serving on " + options.host + " requires TENSORFOLD_API_KEY; only loopback may be open");
    options.max_tokens = std::min(options.max_tokens,options.context);
    return options;
}

inline Socket listen_on(const ServeOptions& options) {
    Socket server(socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0));
    int yes = 1;
    setsockopt(server.fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(uint16_t(options.port));
    if (inet_pton(AF_INET,options.loopback() ? "127.0.0.1" : options.host.c_str(),&address.sin_addr) != 1)
        throw std::invalid_argument("--host must be an IPv4 address or localhost");
    if (bind(server.fd,reinterpret_cast<const sockaddr*>(&address),sizeof(address)) < 0 || listen(server.fd,16) < 0)
        throw std::runtime_error("cannot listen on " + options.host + ":" + std::to_string(options.port));
    return server;
}

// Compares every byte so the time taken does not reveal a matching prefix.
inline bool authorized(const ServeOptions& options, const std::string& authorization) {
    if (options.key.empty()) return true;
    const std::string expected = "Bearer " + options.key;
    unsigned char difference = authorization.size() != expected.size();
    for (size_t i = 0; i < expected.size(); ++i)
        difference |= static_cast<unsigned char>(expected[i] ^ (i < authorization.size() ? authorization[i] : 0));
    return !difference;
}

// One chat completion. Returns the JSON body for a plain reply, or null once a
// stream has been written to the client.
inline Json chat(int fd, const ServeOptions& options, Engine& engine, PrefixCache& cache, TextCodec& codec,
                 const std::string& body, unsigned long serial) {
    ChatRequest request = parse_chat(body,options.model);
    mini_vision::ImagePatches image{};
    if (!request.image.empty()) {
        if (!options.vision) throw Error(400,"start Inkling with --vision for image input");
        try { image = mini_vision::decode(request.image); }
        catch (const std::runtime_error& error) { throw std::invalid_argument(error.what()); }
        for (auto& message : request.messages)
            for (auto& part : message.parts)
                if (part.image) part.rows = size_t(image.rows)*image.cols;
    }
    const std::vector<int> prompt = codec.conversation(request.messages,request.effort,request.tools);
    if (prompt.size() >= size_t(options.context)) throw Error(400,"the conversation does not fit the context window");
    const int room = options.context-int(prompt.size());
    const int budget = std::min({request.max_tokens ? request.max_tokens : options.max_tokens,options.max_tokens,room});
    // Refuse a context the device cannot hold before any of the response is sent.
    try { inkling_admit(engine,prompt.size()+budget); }
    catch (const inkling_context::MemoryShortfall&) { throw Error(503,"not enough free GPU memory for a conversation this long"); }

    if (!request.stream) {
        std::vector<int> ids;
        inkling_generate(engine,prompt,image,budget,[&](int id) { ids.push_back(id); return true; },&cache);
        return completion(ids,codec,options.model,prompt.size(),serial);
    }

    // Streaming. The role chunk goes out before prefill so a proxy sees the
    // response start; a client that hangs up frees the engine at the next token.
    bool connected = begin_events(fd) && send_event(fd,dump(chunk(options.model,serial,{{"role","assistant"},{"content",""}})));
    if (!connected) return nullptr;
    Splitter splitter;
    std::string pending; // bytes of the current channel not yet sent
    Splitter::Channel pending_channel = Splitter::none;
    auto flush = [&](bool everything) {
        const size_t ready_bytes = everything ? pending.size() : complete_utf8(pending);
        if (!ready_bytes || !connected) return;
        const char* field = pending_channel == Splitter::reasoning ? "reasoning_content" : "content";
        connected = send_event(fd,dump(chunk(options.model,serial,{{field,pending.substr(0,ready_bytes)}})));
        pending.erase(0,ready_bytes);
    };
    auto last_event = Clock::now();
    // A tool call is relayed whole once the model has finished printing it.
    size_t relayed = 0;
    auto relay_calls = [&] {
        for (; connected && relayed < splitter.calls.size(); ++relayed) {
            Json call = tool_call(splitter.calls[relayed]);
            call["index"] = relayed;
            connected = send_event(fd,dump(chunk(options.model,serial,{{"tool_calls",Json::array({call})}})));
        }
    };
    int generated = 0;
    try {
        generated = inkling_generate(engine,prompt,image,budget,[&](int id) {
            const std::string_view bytes = splitter.feed(codec,id);
            if (splitter.channel != pending_channel) { flush(true); pending_channel = splitter.channel; }
            pending.append(bytes);
            flush(false);
            relay_calls();
            return connected;
        },&cache,[&] {
            // While a long prompt is read: an empty delta every 15 s keeps proxies from
            // timing the request out, and finds out whether anyone still waits for it.
            if (Clock::now()-last_event < std::chrono::seconds(15)) return true;
            last_event = Clock::now();
            return connected = send_event(fd,dump(chunk(options.model,serial,Json::object())));
        });
    } catch (const PrefillCancelled&) {
        return nullptr;
    } catch (const inkling_context::MemoryShortfall&) {
        if (connected) send_event(fd,dump(Json{{"error",{{"message","not enough free GPU memory for a conversation this long"},{"type","server_error"}}}}));
        return nullptr;
    } catch (const std::exception& error) {
        std::cerr << "Inkling generation: " << error.what() << '\n';
        if (connected) send_event(fd,dump(Json{{"error",{{"message","generation failed"},{"type","server_error"}}}}));
        return nullptr;
    }
    flush(true);
    splitter.finish();
    relay_calls();
    if (connected) connected = send_event(fd,dump(chunk(options.model,serial,Json::object(),splitter.finish_reason())));
    if (connected && request.usage_event) {
        Json tail = envelope("chat.completion.chunk",options.model,serial);
        tail["choices"] = Json::array();
        tail["usage"] = usage(prompt.size(),size_t(generated));
        connected = send_event(fd,dump(tail));
    }
    if (connected) send_event(fd,"[DONE]");
    return nullptr;
}
} // namespace ink_http

inline int inkling_serve(int argc, char** argv) {
    using namespace ink_http;
    const ServeOptions options = serve_options(argc,argv);
    hipDeviceProp_t props{};
    INKLING_HIP_CHECK(hipGetDeviceProperties(&props,0));
    if (std::string(props.gcnArchName).substr(0,7) != "gfx1151") throw std::runtime_error("Inkling requires gfx1151");
    Socket server = listen_on(options);
    TextCodec codec(options.checkpoint);
    Engine engine(options.checkpoint);
    PrefixCache cache;
    std::cerr << "Inkling serving " << options.model << " on http://" << options.host << ":" << options.port
              << (options.key.empty() ? "" : " (API key required)") << std::endl;
    // Without a key the listener is loopback: refuse foreign Host headers so a web page cannot reach it.
    const bool loopback_only = options.key.empty();
    for (unsigned long serial = 0;;) {
        Socket client(accept4(server.fd,nullptr,nullptr,SOCK_CLOEXEC));
        if (client.fd < 0) { if (errno == EINTR) continue; throw std::runtime_error("accept failed"); }
        Json response;
        int status = 200;
        try {
            const Request request = read_request(client.fd,loopback_only);
            if (request.line == "GET /health") response = {{"status","ok"},{"model",options.model}};
            else if (!authorized(options,request.authorization)) throw Error(401,"missing or invalid API key");
            else if (request.line == "GET /v1/models") {
                Json entry = {{"id",options.model},{"object","model"},{"created",0},{"owned_by","local"}};
                response = {{"object","list"},{"data",Json::array({entry})}};
            } else if (request.line == "POST /v1/chat/completions") {
                response = chat(client.fd,options,engine,cache,codec,request.body,++serial);
                if (response.is_null()) continue; // already streamed
            } else throw Error(404,"unknown route or method");
        } catch (const Error& error) { status = error.status; response = error.what(); }
        catch (const inkling_context::MemoryShortfall&) { status = 503; response = "not enough free GPU memory for a conversation this long"; }
        catch (const Json::exception&) { status = 400; response = "invalid chat JSON"; }
        catch (const std::invalid_argument& error) { status = 400; response = error.what(); }
        catch (const std::out_of_range& error) { status = 400; response = error.what(); }
        catch (const std::exception& error) {
            std::cerr << "Inkling generation: " << error.what() << '\n';
            status = 500; response = "generation failed";
        }
        if (status != 200)
            response = {{"error",{{"message",response},{"type",status < 500 ? "invalid_request_error" : "server_error"}}}};
        reply(client.fd,status,dump(response));
    }
}
