// Generate reference.png and reference.f32 with vision_reference.py before running.
#include "vision.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: vision-test CHECKPOINT FIXTURE_DIR");
    std::string dir = argv[2];
    std::ifstream png(dir+"/reference.png",std::ios::binary);
    std::vector<unsigned char> image{std::istreambuf_iterator<char>(png),{}};
    if (image.size()<33) throw std::runtime_error("missing fixture image");
    auto reject=[](const std::vector<unsigned char>& bad) {
        try { mini_vision::decode(bad); } catch(const std::runtime_error&) {return;}
        throw std::runtime_error("accepted invalid PNG");
    };
    auto bad=image; bad.pop_back(); reject(bad); // truncated IEND
    bad=image; bad[29]^=1; reject(bad); // corrupt IHDR CRC
    bad=image; bad[16]=127; // oversized width, but valid IHDR CRC
    unsigned crc=crc32(0,bad.data()+12,17);
    for(int i=0;i<4;++i) bad[29+i]=crc>>(24-8*i);
    reject(bad);
    Engine engine(argv[1]);
    auto actual=inkling_image(engine,image);
    if (actual.size()!=6*2304) throw std::runtime_error("incorrect projector shape");
    std::vector<float> expected(actual.size());
    std::ifstream ref(dir+"/reference.f32",std::ios::binary);
    if (!ref.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(float)) || ref.peek()!=EOF)
        throw std::runtime_error("missing or incorrectly sized HF reference");
    float worst=0;
    for(size_t i=0;i<expected.size();++i){
        float delta=std::abs(actual[i]-expected[i]);
        if (!(delta<=1e-4f)) throw std::runtime_error("HF mismatch at element "+std::to_string(i));
        worst=std::max(worst,delta);
    }
    std::cout << "6 x 2304 projected rows; max absolute error " << worst << '\n';
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
