#pragma once

#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace mini_engine {
using Json = nlohmann::json;
namespace fs = std::filesystem;

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("Inkling Engine: " + message);
}
inline size_t product(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) throw std::overflow_error("Inkling Engine: size overflow");
    return a * b;
}

// Read-only mappings also support HF snapshot symlinks. File contents must
// remain unchanged for the life of the Engine.
struct Mapping {
    size_t size;
    const uint8_t* data;
    explicit Mapping(const fs::path& file) {
        int fd = open(file.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), file.string());
        struct stat st;
        if (fstat(fd, &st) < 0) {
            int err = errno; close(fd);
            throw std::system_error(err, std::generic_category(), file.string());
        }
        if (!S_ISREG(st.st_mode) || st.st_size <= 0 || uintmax_t(st.st_size) > SIZE_MAX) {
            close(fd); throw std::runtime_error("Invalid checkpoint file: " + file.string());
        }
        size = size_t(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        int err = errno; close(fd);
        if (p == MAP_FAILED) throw std::system_error(err, std::generic_category(), file.string());
        data = static_cast<const uint8_t*>(p);
    }
    ~Mapping() { munmap(const_cast<uint8_t*>(data), size); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};

inline Json metadata(const uint8_t* data, size_t size) {
    require(size > 0 && size <= 100000000, "invalid metadata length");
    require(!std::memchr(data, 0, size), "NUL byte in JSON document");
    // The default DOM parser silently overwrites duplicate object keys.
    std::vector<std::set<std::string>> keys;
    return Json::parse(data, data + size, [&](int depth, Json::parse_event_t event, Json& value) {
        require(depth <= 64, "metadata nesting limit exceeded");
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        if (event == Json::parse_event_t::key)
            require(keys.back().insert(value.get<std::string>()).second, "duplicate metadata key");
        if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    });
}
inline Json document(const fs::path& path) {
    Mapping m(path);
    return metadata(m.data, m.size);
}
inline size_t natural(const Json& value) {
    require(value.is_number_unsigned() && value.get<uint64_t>() <= SIZE_MAX, "invalid size");
    return value.get<size_t>();
}

enum class Type { bf16, f32, u8, e4m3 };
struct Tensor { const uint8_t* data; size_t bytes, count; Type type; };

class Checkpoint {
    std::map<std::string, Mapping> shards;
    std::map<std::string, Tensor> tensors;

    void scan(const Mapping& m, const std::string& name, const Json& index) {
        require(m.size >= 8, "truncated shard: " + name);
        uint64_t h = 0;
        for (int i = 0; i < 8; ++i) h |= uint64_t(m.data[i]) << (8*i);
        require(h <= m.size - 8, "truncated shard header: " + name);
        auto header = metadata(m.data + 8, h);
        require(header.is_object(), "expected tensor dictionary");
        size_t base = 8 + h;
        std::vector<std::pair<size_t,size_t>> spans;
        for (const auto& [key, v] : header.items()) {
            if (key == "__metadata__") continue;
            require(index.contains(key) && index.at(key) == name, "tensor/index mismatch: " + key);
            const auto& dtype = v.at("dtype");
            Type type;
            size_t width;
            if (dtype == "BF16") { type = Type::bf16; width = 2; }
            else if (dtype == "F32") { type = Type::f32; width = 4; }
            else if (dtype == "U8") { type = Type::u8; width = 1; }
            else if (dtype == "F8_E4M3") { type = Type::e4m3; width = 1; }
            else throw std::runtime_error("Unsupported dtype for " + key + ": " + dtype.dump());
            size_t count = 1;
            const auto& shape = v.at("shape");
            require(shape.is_array(), "invalid shape: " + key);
            for (const auto& axis : shape) count = product(count, natural(axis));
            const auto& offsets = v.at("data_offsets");
            require(offsets.is_array() && offsets.size() == 2, "invalid offsets: " + key);
            size_t begin = natural(offsets[0]), end = natural(offsets[1]);
            require(begin <= end && end <= m.size-base && end-begin == product(count,width),
                    "out-of-bounds or wrong byte length: " + key);
            spans.emplace_back(begin, end);
            tensors.emplace(key, Tensor{m.data + base + begin, end-begin, count, type});
        }
        std::sort(spans.begin(), spans.end());
        size_t cursor = 0;
        for (auto [begin,end] : spans) {
            require(begin == cursor, "overlap or hole in shard: " + name);
            cursor = end;
        }
        require(cursor == m.size-base, "unindexed trailing bytes: " + name);
    }
public:
    explicit Checkpoint(const fs::path& dir) {
        auto config = document(dir / "config.json");
        const auto& text = config.at("text_config");
        require(config.at("model_type") == "inkling2_loop" &&
                config.at("architectures") == Json::array({"Inkling2LoopForConditionalGeneration"}) &&
                text.at("model_type") == "inkling2_loop_text" && text.at("hidden_size") == 2304 &&
                text.at("num_hidden_layers") == 24 && text.at("vocab_size") == 201024 &&
                config.at("quantization_config").at("quant_algo") == "NVFP4", "unsupported model config");
        auto manifest = document(dir / "model.safetensors.index.json");
        const auto& index = manifest.at("weight_map");
        require(index.is_object() && !index.empty(), "invalid weight_map");
        for (const auto& value : index) {
            auto name = value.get<std::string>();
            require(fs::path(name).filename() == name && fs::path(name).extension() == ".safetensors" &&
                    name.find('\0') == name.npos, "invalid shard filename");
            auto [it, added] = shards.try_emplace(name, dir / name);
            if (added) scan(it->second, name, index);
        }
        require(tensors.size() == index.size(), "weight_map references missing tensors");
    }
    const Tensor& get(const std::string& key) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) throw std::out_of_range("Missing Inkling tensor: " + key);
        return it->second;
    }
};
} // namespace mini_engine
