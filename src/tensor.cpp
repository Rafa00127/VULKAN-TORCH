#include "tensor.h"

#include "ggml-backend.h"
#include "graph.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mt {

std::vector<int64_t> Tensor::shape() const {
    std::vector<int64_t> s;
    if (!t_) return s;
    const int nd = ggml_n_dims(t_);
    s.reserve(nd);
    for (int i = nd - 1; i >= 0; --i) s.push_back(t_->ne[i]);
    return s;
}

int64_t Tensor::numel() const {
    return t_ ? ggml_nelements(t_) : 0;
}

bool Tensor::is_contiguous() const {
    return t_ ? ggml_is_contiguous(t_) : false;
}

void Tensor::mark_output() const {
    if (t_ != nullptr) ggml_set_output(t_);
}

static void require_contiguous(const ggml_tensor* t) {
    if (!ggml_is_contiguous(t)) {
        throw std::runtime_error(
            "Tensor: cannot read a non-contiguous tensor; call contiguous() first");
    }
}

std::vector<float> Tensor::to_host() const {
    if (!t_) throw std::runtime_error("to_host: undefined tensor");
    if (g_ != nullptr) g_->compute_if_needed();
    require_contiguous(t_);

    std::vector<float> out(static_cast<std::size_t>(ggml_nelements(t_)));
    ggml_backend_tensor_get(t_, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::string Tensor::to_host_bytes() const {
    if (!t_) throw std::runtime_error("to_host_bytes: undefined tensor");
    if (g_ != nullptr) g_->compute_if_needed();
    require_contiguous(t_);
    const size_t n = ggml_nbytes(t_);
    std::string out(n, '\0');
    ggml_backend_tensor_get(t_, out.data(), 0, n);
    return out;
}

std::string Tensor::backend_name() const {
    if (!t_ || !t_->buffer) return "<no buffer>";
    return ggml_backend_buffer_name(t_->buffer);
}

std::string Tensor::repr() const {
    if (!t_) return "Tensor(undefined)";
    std::ostringstream os;
    os << "Tensor(shape=[";
    const auto s = shape();
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) os << ", ";
        os << s[i];
    }
    os << "], type=" << ggml_type_name(t_->type)
       << ", contig=" << (ggml_is_contiguous(t_) ? "1" : "0") << ")";
    return os.str();
}

}  // namespace mt
