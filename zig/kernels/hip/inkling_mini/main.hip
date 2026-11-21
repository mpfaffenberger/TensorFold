#include "model.hpp"
#include "tokenizer.hpp"
#include <iostream>
int main(int argc,char** argv) try {
    if(argc<3||argc>4){std::cerr<<"usage: tf-inkling-hip CHECKPOINT PROMPT [MAX_NEW_TOKENS]\n";return 2;}
    hipDeviceProp_t props{}; INKLING_HIP_CHECK(hipGetDeviceProperties(&props,0));
    if(std::string(props.gcnArchName).substr(0,7)!="gfx1151") throw std::runtime_error("requires gfx1151");
    TextCodec codec(argv[1]);auto prompt=codec.turn(argv[2]); int limit=argc==4?std::stoi(argv[3]):96;
    if(limit<1 || limit>4096 || prompt.size()+limit>4096) throw std::runtime_error("experimental CLI limited to 4096 positions");
    Engine engine(argv[1]);std::cerr<<props.gcnArchName<<", "<<prompt.size()<<" prompt tokens\n";int pos=0,next=0;
    for(int id:prompt)next=decode(engine,id,pos++);
    const bool ids = std::getenv("TF_INKLING_TOKEN_IDS") != nullptr;
    for(int n=0;n<limit;++n){
        if(ids) std::cout << next << ' '; else std::cout << codec.word(next);
        std::cout.flush();
        if(next==200006 || n+1==limit)break;
        next=decode(engine,next,pos++);
    }
    std::cout<<'\n';return 0;
} catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
