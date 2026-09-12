#pragma once

#include "ggml.h"

#include "runtime.h"
#include "tensor.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;

namespace vt {

class Memory;

// Minimal GGUF reader. Each tensor is loaded into a device-resident Memory,
// keeping its native ggml type (quantized weights stay quantized).
class GgufFile {
public:
    GgufFile(const std::string& path, Device dev, size_t arena_bytes);
    ~GgufFile();

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    std::vector<std::string> names() const;
    int count() const;
    const char* name_at(int i) const;  // valid for the lifetime of the file
    bool has(const std::string& name) const;
    std::vector<int64_t> shape(const std::string& name) const;  // PyTorch order
    std::string type_name(const std::string& name) const;
    size_t nbytes(const std::string& name) const;

    Tensor tensor(const std::string& name);  // copy into Memory(device)

private:
    ggml_tensor* find(const std::string& name) const;

    std::string path_;
    Device dev_;
    gguf_context* gguf_ = nullptr;
    ggml_context* meta_ = nullptr;  // metadata-only tensors (no data)
    // weights may exceed the Vulkan per-buffer limit, so they are spread across
    // multiple arena buffers (like ggml_backend_alloc_ctx_tensors chunking)
    std::vector<std::shared_ptr<Memory>> mems_;
    size_t arena_cap_ = 0;
    size_t data_offset_ = 0;
};

}  // namespace vt
