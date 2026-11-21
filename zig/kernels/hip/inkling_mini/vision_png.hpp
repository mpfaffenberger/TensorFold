#pragma once
#include <zlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mini_vision {
struct ImagePatches { int rows, cols; std::vector<float> values; };
inline void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline unsigned be32(const unsigned char* p) {
    return (unsigned(p[0])<<24) | (unsigned(p[1])<<16) | (unsigned(p[2])<<8) | p[3];
}
inline int paeth(int a,int b,int c) {
    int p=a+b-c, x=abs(p-a), y=abs(p-b), z=abs(p-c);
    return x<=y && x<=z ? a : y<=z ? b : c;
}
// Deliberately narrow PNG profile. Never allocate based on a compressed chunk length alone.
inline ImagePatches decode(const std::vector<unsigned char>& bytes) {
    check(bytes.size()>=33 && bytes.size()<=8*1024*1024 &&
          !memcmp(bytes.data(),"\x89PNG\r\n\x1a\n",8), "expected PNG (8 MiB limit)");
    int w=0,h=0,ch=0,rows=0,cols=0;
    std::vector<unsigned char> compressed;
    bool ended=false, data=false, after_data=false;
    for(size_t off=8; bytes.size()-off>=12;) {
        auto p=bytes.data()+off; auto len=be32(p); auto tag=p+4; auto v=p+8;
        check(len<=bytes.size()-off-12, "truncated PNG chunk");
        check(crc32(0,tag,len+4)==be32(v+len), "invalid PNG CRC");
        if(!memcmp(tag,"IHDR",4)) {
            check(off==8 && len==13,"invalid PNG header");
            unsigned width=be32(v),height=be32(v+4);
            check(width && height && width<=10240 && height<=10240,"PNG dimensions exceed limit");
            w=width; h=height; rows=(h+39)/40; cols=(w+39)/40;
            check(rows*cols<=256,"PNG exceeds 256 merged tokens");
            check(v[8]==8 && !v[10] && !v[11] && !v[12],"expected non-interlaced 8-bit PNG");
            switch(v[9]) {case 0: ch=1; break; case 2: ch=3; break; case 4: ch=2; break;
                case 6: ch=4; break; default: throw std::runtime_error("unsupported PNG color type");}
        } else {
            check(ch!=0,"missing PNG header");
            if(!memcmp(tag,"IDAT",4)) {
                check(!after_data,"noncontiguous PNG data"); data=true;
                compressed.insert(compressed.end(),v,v+len);
            } else if(!memcmp(tag,"IEND",4)) {
                check(len==0 && off+12==bytes.size() && data,"invalid PNG end"); ended=true; break;
            } else {
                check((tag[0]&32) && memcmp(tag,"tRNS",4) && memcmp(tag,"acTL",4),"unsupported PNG chunk");
                after_data=data;
            }
        }
        off+=12+len;
    }
    check(ended,"missing PNG end");
    int stride=1+w*ch;
    std::vector<unsigned char> raw(h*stride);
    uLongf dest=raw.size(); uLong src=compressed.size();
    check(uncompress2(raw.data(),&dest,compressed.data(),&src)==Z_OK && dest==raw.size() && src==compressed.size(),"invalid PNG data");
    ImagePatches out{rows,cols,std::vector<float>(rows*cols*4800,-1.f-2.f/255.f)};
    for(int y=0;y<h;++y) {
        auto row=raw.data()+y*stride+1; int filter=row[-1]; check(filter<=4,"invalid PNG filter");
        for(int i=0;i<w*ch;++i) {
            int a=i>=ch?row[i-ch]:0,b=y?row[i-stride]:0,c=y&&i>=ch?row[i-stride-ch]:0;
            row[i]+=filter==0?0:filter==1?a:filter==2?b:filter==3?(a+b)/2:paeth(a,b,c);
        }
        for(int x=0;x<w;++x) {
            int patch=((y/40)*cols+x/40)*4+(y/20%2)*2+x/20%2;
            auto pixel=row+x*ch; int alpha=ch%2?255:pixel[ch-1], bg=(x/8+y/8)%2?153:204;
            for(int k=0;k<3;++k) {
                int rgb=(pixel[ch<=2?0:k]*alpha+bg*(255-alpha)+127)/255;
                out.values[((patch*20+y%20)*20+x%20)*3+k]=rgb*(2.f/255.f)-1.f;
            }
        }
    }
    return out;
}
}
