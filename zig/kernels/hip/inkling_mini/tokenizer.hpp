#pragma once
// Single-user text template. Use the checkpoint's byte ranks and pretokenizer regex.
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <fstream>
#include <unordered_map>
#include "engine_checkpoint.hpp"

class TextCodec {
    std::unordered_map<std::string,int> ranks, special;
    std::vector<std::string> words{201024};
    pcre2_code* code = nullptr;
    static std::string unbase64(const std::string& src) {
        static const std::string chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        unsigned acc=0; int bits=0; std::string out;
        for(char c:src){if(c=='=')break; auto k=chars.find(c); if(k==std::string::npos)throw std::runtime_error("invalid base64"); acc=(acc<<6)|k; bits+=6; if(bits>=8){bits-=8;out.push_back(static_cast<char>(acc>>bits));}} return out;
    }
    void piece(std::vector<int>& out,std::string s) const {
        if(auto it=ranks.find(s);it!=ranks.end()){out.push_back(it->second);return;}
        std::vector<std::string> parts; for(char c:s)parts.emplace_back(1,c);
        while(parts.size()>1){int best=INT_MAX; size_t at=0;for(size_t j=1;j<parts.size();++j){auto it=ranks.find(parts[j-1]+parts[j]);if(it!=ranks.end()&&it->second<best){at=j;best=it->second;}}if(best==INT_MAX)break;parts[at-1]+=parts[at];parts.erase(parts.begin()+at);}
        for(auto& p:parts)out.push_back(ranks.at(p));
    }
    void text(std::vector<int>& ids,const std::string& s) const {
        std::unique_ptr<pcre2_match_data,decltype(&pcre2_match_data_free)> data(pcre2_match_data_create_from_pattern(code,nullptr),pcre2_match_data_free);
        for(size_t at=0;at<s.size();){if(pcre2_match(code,(PCRE2_SPTR)s.data(),s.size(),at,PCRE2_ANCHORED,data.get(),nullptr)<1)throw std::runtime_error("token split failed");auto end=pcre2_get_ovector_pointer(data.get())[1]; if(end<=at)throw std::runtime_error("empty split");piece(ids,s.substr(at,end-at));at=end;}
    }
public:
    explicit TextCodec(const std::string& dir){std::ifstream f(dir+"/tiktoken/tokenizer.model"); std::string b; int id;while(f>>b>>id){auto w=unbase64(b);words.at(id)=w;ranks.emplace(w,id);} if(ranks.size()<190000)throw std::runtime_error("missing tokenizer ranks");
        std::ifstream json(dir+"/tokenizer.json"); const auto cfg=nlohmann::json::parse(json);for(auto& t:cfg.at("added_tokens")){std::string s=t.at("content");int n=t.at("id");special[s]=n;words.at(n)=s;}std::string pattern=cfg.at("pre_tokenizer").at("pretokenizers").at(0).at("pattern").at("Regex");int err;size_t off;code=pcre2_compile((PCRE2_SPTR)pattern.data(),pattern.size(),PCRE2_UTF|PCRE2_UCP,&err,&off,nullptr);if(!code)throw std::runtime_error("invalid tokenizer regex");}
    ~TextCodec(){pcre2_code_free(code);}
    const std::string& word(int id)const{return words.at(id);}
    std::vector<int> turn(const std::string& prompt)const{ std::vector<int> ids;auto tag=[&](const char*s){ids.push_back(special.at(s));};tag("<|message_system|>");tag("<|content_text|>");text(ids,"Thinking effort level: 0");tag("<|end_message|>");tag("<|message_user|>");tag("<|content_text|>");text(ids,prompt);tag("<|end_message|>");return ids;}
};
