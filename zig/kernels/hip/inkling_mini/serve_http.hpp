#pragma once
// One fixed-length HTTP/1.x request per loopback connection; no worker queue in userspace.
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <algorithm>
#include <cctype>

namespace ink_http {
struct Error : std::runtime_error {
    int status;
    Error(int code, const char* message) : std::runtime_error(message), status(code) {}
};
struct Socket {
    int fd;
    explicit Socket(int n) : fd(n) {}
    ~Socket() { if (fd >= 0) close(fd); }
    Socket(const Socket&) = delete;
};
using Clock = std::chrono::steady_clock;
inline void ready(int fd, short event, Clock::time_point end) {
    pollfd p{fd,event,0};
    for (;;) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-Clock::now()).count();
        if (ms <= 0) throw Error(408,"connection deadline exceeded");
        int n = poll(&p,1,static_cast<int>(ms));
        if (n > 0) return;
        if (n == 0 || errno != EINTR) throw Error(408,"connection unavailable");
    }
}
inline size_t number(std::string_view s, size_t cap) {
    size_t n = 0;
    auto r = std::from_chars(s.data(),s.data()+s.size(),n);
    if (r.ec != std::errc{} || r.ptr != s.data()+s.size()) throw Error(400,"invalid integer");
    if (n > cap) throw Error(413,"limit exceeded");
    return n;
}
inline void reply(int fd, int code, const std::string& body) {
    const std::string wire = "HTTP/1.1 " + std::to_string(code) + (code==200?" OK":" Error") +
        "\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    const auto end = Clock::now()+std::chrono::seconds(5);
    for (size_t off=0; off<wire.size();) {
        ready(fd,POLLOUT,end);
        auto n=send(fd,wire.data()+off,wire.size()-off,MSG_DONTWAIT|MSG_NOSIGNAL);
        if (n<0 && (errno==EINTR || errno==EAGAIN)) continue;
        if (n<=0) return;
        off+=size_t(n);
    }
}
struct Request { std::string line, body; };
inline Request read_request(int fd) {
    constexpr size_t headers_cap=16384, body_cap=8*1024*1024;
    const auto end=Clock::now()+std::chrono::seconds(10);
    std::string buf;
    auto more=[&](size_t cap) {
        char chunk[8192];
        for (;;) {
            ready(fd,POLLIN,end);
            auto n=recv(fd,chunk,std::min(cap,sizeof(chunk)),MSG_DONTWAIT);
            if (n<0 && (errno==EINTR || errno==EAGAIN)) continue;
            if (n<=0) throw Error(400,"incomplete request");
            buf.append(chunk,size_t(n)); return;
        }
    };
    size_t split;
    while ((split=buf.find("\r\n\r\n"))==std::string::npos) {
        if (buf.size()>=headers_cap) throw Error(431,"headers too large");
        more(headers_cap-buf.size());
    }
    size_t pos=buf.find("\r\n");
    auto line=buf.substr(0,pos);
    auto ver=line.rfind(' ');
    if (ver==line.npos || (line.substr(ver+1)!="HTTP/1.1" && line.substr(ver+1)!="HTTP/1.0")) throw Error(400,"invalid request line");
    line.resize(ver);
    size_t length=0; bool found=false;
    for (pos+=2; pos<split; ) {
        auto e=buf.find("\r\n",pos);
        auto h=buf.substr(pos,e-pos); pos=e+2;
        auto k=h.find(':');
        if (!k || k==h.npos || h.find_first_of(" \t\r\n")<k) throw Error(400,"invalid header");
        auto name=h.substr(0,k);
        for (char& c:name) c=char(std::tolower(static_cast<unsigned char>(c)));
        auto value=h.substr(k+1); auto first=value.find_first_not_of(" \t");
        value=first==value.npos?"":value.substr(first,value.find_last_not_of(" \t")-first+1);
        if (name=="transfer-encoding") throw Error(400,"chunked requests unsupported");
        if (name=="content-length") {
            if (found) throw Error(400,"duplicate content length");
            found=true; length=number(value,body_cap);
        }
        if (name=="origin" && value!="") throw Error(403,"browser origins not supported");
        if (name=="host" && value.substr(0,value.find(':'))!="127.0.0.1" && value.substr(0,value.find(':'))!="localhost") throw Error(403,"loopback host required");
        if (name=="content-type" && value.substr(0,value.find(';'))!="application/json") throw Error(415,"expected application/json");
    }
    buf.erase(0,split+4);
    if (buf.size()>length) throw Error(400,"unexpected trailing bytes");
    while (buf.size()<length) more(length-buf.size());
    return {line,std::move(buf)};
}
} // namespace ink_http
