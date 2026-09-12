#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "runtime.h"
#include "tensor.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace vt {

// Persistent, device-resident storage for weights: one ggml context plus one
// backend buffer (WEIGHTS usage) bound to a Device. Tensors created here live on
// that device; ggml runs an op on the backend of its weight, so consuming ops
// execute on the same device (CPU inputs are copied over as needed).
//
// Tensors returned by tensor() keep this Memory alive, so a temporary
// `mt.Memory(...).tensor(...)` is safe, like torch storage.
class Memory : public std::enable_shared_from_this<Memory> {
public:
    Memory(Device dev, size_t max_bytes);
    ~Memory();

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Device device() const { return dev_; }
    ggml_context* ctx() const { return ctx_; }
    size_t remaining() const { return cap_ - offset_; }

    Tensor tensor(const std::vector<int64_t>& pt_shape);  // uninitialized (F32)
    Tensor tensor(const std::vector<int64_t>& pt_shape, const void* data, size_t bytes);
    Tensor tensor(const std::vector<int64_t>& pt_shape, ggml_type type, const void* data,
                  size_t bytes);

private:
    Device dev_;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    size_t cap_ = 0;
    size_t offset_ = 0;
};

}  // namespace vt
