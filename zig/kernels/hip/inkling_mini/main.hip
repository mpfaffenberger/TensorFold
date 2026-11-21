#include "model.hpp"
#include "tokenizer.hpp"
#include "generate.hpp"
#include "serve.hpp"
#include <iostream>
int main(int argc,char** argv) try {
    if (argc >= 2 && std::string(argv[1]) == "serve") return inkling_serve(argc, argv);
    if(argc<3||argc>4){std::cerr<<"usage: tf-inkling-hip CHECKPOINT PROMPT [MAX_NEW_TOKENS]\n";return 2;}
    hipDeviceProp_t props{}; INKLING_HIP_CHECK(hipGetDeviceProperties(&props,0));
    if(std::string(props.gcnArchName).substr(0,7)!="gfx1151") throw std::runtime_error("requires gfx1151");
    TextCodec codec(argv[1]);int limit=argc==4?std::stoi(argv[3]):96;
    Engine engine(argv[1]);
    auto ids=inkling_generate(engine,codec,argv[2],{},limit,16384);
    for(int id:ids){if(std::getenv("TF_INKLING_TOKEN_IDS")) std::cout<<id<<' '; else std::cout<<codec.word(id);}
    std::cout<<'\n';return 0;
} catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
