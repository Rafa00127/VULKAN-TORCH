#include "gguf_file.h"

#include "ggml.h"
#include "gguf.h"

#include "memory.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace mt {

GgufFile::GgufFile(const std::string& path, Device dev, size_t arena_bytes)
    : path_(path), dev_(dev) {
    gguf_init_params gp = {true /*no_alloc*/, &meta_};
    gguf_ = gguf_init_from_file(path.c_str(), gp);
    if (gguf_ == nullptr || meta_ == nullptr) {
        throw std::runtime_error("GgufFile: failed to open " + path);
    }
    data_offset_ = gguf_get_data_offset(gguf_);
    // Chunk arenas so no single Vulkan buffer exceeds the device limit.
    const size_t max_alloc = ggml_backend_get_max_size(dev.backend());
    arena_cap_ = (max_alloc > 0 && max_alloc < arena_bytes) ? max_alloc : arena_bytes;
}

GgufFile::~GgufFile() {
    if (gguf_ != nullptr) gguf_free(gguf_);
    if (meta_ != nullptr) ggml_free(meta_);
}

ggml_tensor* GgufFile::find(const std::string& name) const {
    return ggml_get_tensor(meta_, name.c_str());
}

std::vector<std::string> GgufFile::names() const {
    std::vector<std::string> out;
    const int64_t n = gguf_get_n_tensors(gguf_);
    out.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out.emplace_back(gguf_get_tensor_name(gguf_, i));
    return out;
}

bool GgufFile::has(const std::string& name) const { return find(name) != nullptr; }

int GgufFile::count() const { return static_cast<int>(gguf_get_n_tensors(gguf_)); }

const char* GgufFile::name_at(int i) const { return gguf_get_tensor_name(gguf_, i); }

std::vector<int64_t> GgufFile::shape(const std::string& name) const {
    ggml_tensor* t = find(name);
    if (t == nullptr) throw std::runtime_error("GgufFile: no tensor " + name);
    const int nd = ggml_n_dims(t);
    std::vector<int64_t> s;
    s.reserve(nd);
    for (int i = nd - 1; i >= 0; --i) s.push_back(t->ne[i]);
    return s;
}

std::string GgufFile::type_name(const std::string& name) const {
    ggml_tensor* t = find(name);
    if (t == nullptr) throw std::runtime_error("GgufFile: no tensor " + name);
    return ggml_type_name(t->type);
}

size_t GgufFile::nbytes(const std::string& name) const {
    ggml_tensor* t = find(name);
    if (t == nullptr) throw std::runtime_error("GgufFile: no tensor " + name);
    return ggml_nbytes(t);
}

Tensor GgufFile::tensor(const std::string& name) {
    ggml_tensor* t = find(name);
    if (t == nullptr) throw std::runtime_error("GgufFile: no tensor " + name);

    const size_t bytes = ggml_nbytes(t);
    const size_t need = bytes + 16;
    if (mems_.empty() || mems_.back()->remaining() < need) {
        mems_.push_back(std::make_shared<Memory>(dev_, std::max(arena_cap_, need)));
    }
    std::vector<int64_t> pt_shape = shape(name);
    Tensor out = mems_.back()->tensor(pt_shape, t->type, nullptr, 0);

    const int64_t id = gguf_find_tensor(gguf_, name.c_str());
    const size_t offset = data_offset_ + gguf_get_tensor_offset(gguf_, id);

    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("GgufFile: cannot reopen " + path_);
    in.seekg(static_cast<std::streamoff>(offset));

    constexpr size_t CHUNK = 1u << 20;  // 1 MiB
    std::vector<char> buf(std::min(CHUNK, bytes));
    for (size_t off = 0; off < bytes; off += buf.size()) {
        const size_t n = std::min(buf.size(), bytes - off);
        in.read(buf.data(), static_cast<std::streamsize>(n));
        if (static_cast<size_t>(in.gcount()) != n) {
            throw std::runtime_error("GgufFile: short read for " + name);
        }
        ggml_backend_tensor_set(out.raw(), buf.data(), off, n);
    }
    return out;
}

}  // namespace mt
