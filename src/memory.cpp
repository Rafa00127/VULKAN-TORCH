#include "memory.h"

#include "graph.h"  // new_tensor_pt

#include <cstdint>
#include <stdexcept>

namespace mt {

Memory::Memory(Device dev, size_t max_bytes) : dev_(dev) {
    // Allocating a backend buffer while a graph capture is active corrupts the
    // scheduler's state; make the mistake an explicit error instead of a crash.
    if (Graph::current() != nullptr) {
        throw std::runtime_error(
            "Memory: cannot create a Memory while a Graph capture is active; "
            "create device-resident weights before entering the capture");
    }

    ggml_init_params params = {ggml_tensor_overhead() * 8192, nullptr, /*no_alloc=*/true};
    ctx_ = ggml_init(params);
    if (ctx_ == nullptr) throw std::runtime_error("Memory: ggml_init failed");

    buf_ = ggml_backend_alloc_buffer(dev.backend(), max_bytes);
    if (buf_ == nullptr) throw std::runtime_error("Memory: ggml_backend_alloc_buffer failed");
    ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    cap_ = max_bytes;
}

Memory::~Memory() {
    if (buf_ != nullptr) ggml_backend_buffer_free(buf_);
    if (ctx_ != nullptr) ggml_free(ctx_);
}

Tensor Memory::tensor(const std::vector<int64_t>& pt_shape) {
    return tensor(pt_shape, GGML_TYPE_F32, nullptr, 0);
}

Tensor Memory::tensor(const std::vector<int64_t>& pt_shape, ggml_type type, const void* data,
                      size_t bytes) {
    ggml_tensor* t = new_tensor_pt(ctx_, type, pt_shape);

    auto* base = static_cast<uint8_t*>(ggml_backend_buffer_get_base(buf_));
    offset_ = (offset_ + 15) & ~static_cast<size_t>(15);  // 16-byte alignment
    if (ggml_backend_tensor_alloc(buf_, t, base + offset_) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Memory: ggml_backend_tensor_alloc failed");
    }
    offset_ += ggml_nbytes(t);

    Tensor out(t, nullptr);
    if (auto self = weak_from_this().lock()) out.keep_alive(self);
    if (data != nullptr) ggml_backend_tensor_set(t, data, 0, bytes);
    return out;
}

Tensor Memory::tensor(const std::vector<int64_t>& pt_shape, const void* data, size_t bytes) {
    return tensor(pt_shape, GGML_TYPE_F32, data, bytes);
}

}  // namespace mt
