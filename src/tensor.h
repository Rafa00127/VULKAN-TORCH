#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vt {

class Graph;

// A handle to a ggml tensor node. Shapes are presented in PyTorch order (the
// reverse of ggml's ne[] order): a PyTorch [d0, d1, ..., dn-1] tensor is backed
// by a ggml tensor whose ne = [dn-1, ..., d1, d0].
//
// `graph() == nullptr` means the tensor is not owned by a capture (e.g. a model
// weight that already lives in a persistent backend buffer).
class Tensor {
public:
    Tensor() = default;
    Tensor(ggml_tensor* t, Graph* g) : t_(t), g_(g) {}

    // Keep an owning storage (e.g. a Memory) alive for as long as this tensor.
    void keep_alive(std::shared_ptr<void> s) { keepalive_ = std::move(s); }

    bool defined() const { return t_ != nullptr; }
    ggml_tensor* raw() const { return t_; }
    Graph* graph() const { return g_; }

    int dim() const { return t_ ? ggml_n_dims(t_) : 0; }
    std::vector<int64_t> shape() const;
    int64_t numel() const;
    bool is_contiguous() const;
    ggml_type dtype() const { return t_ ? t_->type : GGML_TYPE_F32; }

    // Materialize to host memory. If this tensor belongs to a capture, the
    // pending graph is computed first.
    std::vector<float> to_host() const;
    std::string to_host_bytes() const;  // raw bytes, avoids per-element conversion
    // Raw readback into a caller buffer; works for any dtype and any tensor that
    // already has a backend buffer (weights included — no graph needed).
    void to_host_bytes_into(void* out, size_t bytes) const;

    // Mark as a graph output so the scheduler will not reuse its buffer. Call
    // this on any tensor you will read AFTER the graph is built (before the
    // first to_host/to_bytes), otherwise its storage may be overwritten.
    void mark_output() const;

    std::string repr() const;

    // Debug: name of the buffer this tensor landed in (CPU vs Vulkan).
    std::string backend_name() const;

private:
    ggml_tensor* t_ = nullptr;
    Graph* g_ = nullptr;
    std::shared_ptr<void> keepalive_;
};

}  // namespace vt
