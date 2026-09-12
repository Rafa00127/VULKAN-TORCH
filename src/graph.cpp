#include "graph.h"

#include "runtime.h"
#include "tensor.h"

#include <algorithm>
#include <stdexcept>

namespace vt {

ggml_tensor* new_tensor_pt(ggml_context* ctx, ggml_type type,
                           const std::vector<int64_t>& pt_shape) {
    std::vector<int64_t> ne(pt_shape.rbegin(), pt_shape.rend());
    return ggml_new_tensor(ctx, type, static_cast<int>(ne.size()), ne.data());
}

int64_t pt_numel(const std::vector<int64_t>& pt_shape) {
    int64_t n = 1;
    for (int64_t s : pt_shape) n *= s;
    return n;
}

static thread_local Graph* t_current = nullptr;
// The graph whose allocation the scheduler currently holds. The scheduler can
// only track one graph at a time, so it is reset when that graph goes away.
static Graph* t_last_computed = nullptr;

Graph* Graph::current() { return t_current; }

void Graph::enter() {
    prev_ = t_current;
    t_current = this;
}

void Graph::exit() { t_current = prev_; }

Graph::Graph(ggml_backend_t backend, ggml_backend_sched_t sched, size_t max_nodes)
    : backend_(backend), sched_(sched), device_(backend, "device") {
    const size_t bytes = ggml_tensor_overhead() * max_nodes +
                         ggml_graph_overhead_custom(max_nodes, false);
    owned_meta_.resize(bytes);
    init(owned_meta_.data(), bytes, max_nodes);
}

Graph::Graph(ggml_backend_t backend, ggml_backend_sched_t sched, void* meta_buf, size_t meta_bytes,
             size_t max_nodes)
    : backend_(backend), sched_(sched), device_(backend, "device") {
    init(meta_buf, meta_bytes, max_nodes);
}

Graph::Graph(Runtime& rt, size_t max_nodes) : Graph(rt, rt.gpu(), max_nodes) {}

Graph::Graph(Runtime& rt, Device dev, size_t max_nodes)
    : backend_(dev.backend()), sched_(rt.sched()), device_(dev) {
    const size_t bytes = ggml_tensor_overhead() * max_nodes +
                         ggml_graph_overhead_custom(max_nodes, false);
    owned_meta_.resize(bytes);
    init(owned_meta_.data(), bytes, max_nodes);
}

void Graph::init(void* meta_buf, size_t meta_bytes, size_t max_nodes) {
    ggml_init_params params = {meta_bytes, meta_buf, /*no_alloc=*/true};
    ctx_ = ggml_init(params);
    if (ctx_ == nullptr) throw std::runtime_error("Graph: ggml_init failed");
    gf_ = ggml_new_graph_custom(ctx_, max_nodes, /*grads=*/false);
    if (gf_ == nullptr) throw std::runtime_error("Graph: ggml_new_graph_custom failed");
}

Graph::~Graph() {
    if (t_last_computed == this) {
        ggml_backend_sched_reset(sched_);
        t_last_computed = nullptr;
    }
    if (ctx_ != nullptr) ggml_free(ctx_);
}

Tensor Graph::input(const std::vector<int64_t>& pt_shape) {
    ggml_tensor* t = new_tensor_pt(ctx_, GGML_TYPE_F32, pt_shape);
    ggml_set_input(t);
    return Tensor(t, this);
}

Tensor Graph::input(const std::vector<int64_t>& pt_shape, const void* data, size_t bytes) {
    ggml_tensor* t = new_tensor_pt(ctx_, GGML_TYPE_F32, pt_shape);
    ggml_set_input(t);
    set_input(t, data, bytes);
    return Tensor(t, this);
}

Tensor Graph::input(const std::vector<int64_t>& pt_shape, const std::vector<float>& data) {
    return input(pt_shape, data.data(), data.size() * sizeof(float));
}

Tensor Graph::input_i32(const std::vector<int64_t>& pt_shape, const void* data, size_t bytes) {
    ggml_tensor* t = new_tensor_pt(ctx_, GGML_TYPE_I32, pt_shape);
    ggml_set_input(t);
    set_input(t, data, bytes);
    return Tensor(t, this);
}

void Graph::set_input(ggml_tensor* node, const void* data, size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    std::copy_n(static_cast<const uint8_t*>(data), bytes, buf.data());
    inputs_.emplace_back(node, std::move(buf));
}

void Graph::compute_if_needed() {
    if (computed_) return;

    ggml_backend_sched_reset(sched_);

    // Placement is driven by the tensors themselves (torch-style): weights live
    // on a device via Memory, and ggml runs each op on the backend of its
    // weight. Graph inputs are copied over as needed. op_offload lets ops the
    // device does not support fall back to CPU.
    if (!ggml_backend_sched_alloc_graph(sched_, gf_)) {
        throw std::runtime_error("Graph: ggml_backend_sched_alloc_graph failed");
    }
    for (auto& [node, data] : inputs_) {
        ggml_backend_tensor_set(node, data.data(), 0, data.size());
    }

    const ggml_status st = ggml_backend_sched_graph_compute(sched_, gf_);
    if (st != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Graph: ggml_backend_sched_graph_compute failed");
    }
    computed_ = true;
    t_last_computed = this;
}

void Graph::read(ggml_tensor* node, void* dst, size_t bytes) {
    compute_if_needed();
    ggml_backend_tensor_get(node, dst, 0, bytes);
}

}  // namespace vt
