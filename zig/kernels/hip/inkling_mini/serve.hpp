#pragma once
// Included by main.hip after Engine and TextCodec. No separate engine per request.
#include "serve_chat.hpp"
#include <iostream>
#include <cstdlib>

std::vector<int> inkling_generate(Engine&, TextCodec&, const std::string&, const std::vector<unsigned char>&, int, int);

inline int inkling_serve(int argc,char** argv) {
    using namespace ink_http;
    if (argc<3) throw std::invalid_argument("usage: tf-inkling-hip serve CHECKPOINT [--context N] [--port N] [--max-tokens N] [--name MODEL]");
    if (const char* key=std::getenv("TENSORFOLD_API_KEY"); key && *key) throw std::invalid_argument("Inkling loopback server does not support API keys");
    int port=8080,context=4096,cap=4096;
    bool vision=false;
    std::string model="Inkling-2-Mini";
    for (int i=3;i<argc;++i) {
        if (std::string_view(argv[i])=="--vision") { vision=true; continue; }
        if (i+1>=argc) throw std::invalid_argument("missing serve flag value");
        std::string k=argv[i], v=argv[++i];
        if (k=="--port") port=number(v,65535);
        else if (k=="--context") context=number(v,16384);
        else if (k=="--max-tokens") cap=number(v,16384);
        else if (k=="--name") model=v;
        else if (k=="--host" && (v=="127.0.0.1" || v=="localhost")) continue;
        else throw std::invalid_argument("unsupported serve flag or value: "+k);
    }
    if (!port || !context || !cap || model.empty()) throw std::invalid_argument("port, context, token cap and model name must be nonempty/nonzero");
    cap=std::min(cap,context);
    hipDeviceProp_t props{}; INKLING_HIP_CHECK(hipGetDeviceProperties(&props,0));
    if (std::string(props.gcnArchName).substr(0,7)!="gfx1151") throw std::runtime_error("Inkling requires gfx1151");
    Socket server(socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0));
    int yes=1; setsockopt(server.fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK); addr.sin_port=htons(port);
    if (bind(server.fd,reinterpret_cast<const sockaddr*>(&addr),sizeof(addr))<0) throw std::runtime_error("cannot bind Inkling loopback listener");
    TextCodec codec(argv[2]); Engine engine(argv[2]);
    if (listen(server.fd,4)<0) throw std::runtime_error("cannot listen on loopback");
    std::cerr<<"Inkling serving "<<model<<" on http://127.0.0.1:"<<port<<"\n";
    unsigned long serial=0;
    for (;;) {
        Socket client(accept(server.fd,nullptr,nullptr));
        if (client.fd<0) { if (errno==EINTR) continue; throw std::runtime_error("accept failed"); }
        Json response; int status=200;
        try {
            auto req=read_request(client.fd);
            if (req.line=="GET /health") response={{"status","ok"},{"model",model}};
            else if (req.line=="GET /v1/models") {
                Json entry={{"id",model},{"object","model"},{"created",0},{"owned_by","local"}};
                response={{"object","list"},{"data",Json::array({entry})}};
            } else if (req.line=="POST /v1/chat/completions") {
                auto t=turn(req.body,model,cap);
                if (!vision && !t.image.empty()) throw Error(400,"start Inkling with --vision for image input");
                auto ids=inkling_generate(engine,codec,t.prompt,t.image,t.tokens,context);
                response=completion(ids,codec,model,t.tokens,++serial);
            } else throw Error(404,"unknown route or method");
        } catch (const Error& e) { status=e.status; response=e.what(); }
        catch (const Json::exception&) { status=400; response="invalid chat JSON"; }
        catch (const std::invalid_argument& e) { status=400; response=e.what(); }
        catch (const std::out_of_range& e) { status=400; response=e.what(); }
        catch (const std::exception& e) { std::cerr<<"Inkling generation: "<<e.what()<<'\n'; status=500; response="generation failed"; }
        if (status!=200) response={{"error",{{"message",response},{"type",status<500?"invalid_request_error":"server_error"}}}};
        try { reply(client.fd,status,response.dump(-1,' ',false,Json::error_handler_t::replace)); }
        catch (const Error&) {} // Slow/disconnected client; the resident engine remains available.
    }
}
