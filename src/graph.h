#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "runtime.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vt {

class Tensor;
class Runtime;

// Create a tensor with a PyTorch-order shape (stored reversed as ggml ne[]).
ggml_tensor* new_tensor_pt(ggml_context* ctx, ggml_type type, const std::vector<int64_t>& pt_shape);
int64_t pt_numel(const std::vector<int64_t>& pt_shape);

// A capture session: ops append nodes to one ggml graph; nothing runs until a
// tensor is materialized (see Graph::compute_if_needed).
class Graph {
public:
    Graph(ggml_backend_t backend, ggml_backend_sched_t sched, size_t max_nodes = 16384);
    Graph(ggml_backend_t backend, ggml_backend_sched_t sched, void* meta_buf, size_t meta_bytes,
          size_t max_nodes = 16384);
    Graph(Runtime& rt, size_t max_nodes = 16384);
    Graph(Runtime& rt, Device dev, size_t max_nodes = 16384);
    ~Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    ggml_context* ctx() const { return ctx_; }
    ggml_cgraph* cgraph() const { return gf_; }
    ggml_backend_t backend() const { return backend_; }
    const Device& device() const { return device_; }

    // Grow the forward graph to include `node` and its dependencies.
    void add(ggml_tensor* node) { ggml_build_forward_expand(gf_, node); }

    // Create an F32 input node (PyTorch-order shape); attach host data later.
    Tensor input(const std::vector<int64_t>& pt_shape);
    // Create an F32 input node with host data uploaded at compute time.
    Tensor input(const std::vector<int64_t>& pt_shape, const void* data, size_t bytes);
    Tensor input(const std::vector<int64_t>& pt_shape, const std::vector<float>& data);
    // I32 input (e.g. get_rows indices).
    Tensor input_i32(const std::vector<int64_t>& pt_shape, const void* data, size_t bytes);

    void set_input(ggml_tensor* node, const void* data, size_t bytes);
    void mark_output(ggml_tensor* node) { ggml_set_output(node); }

    // Static replay path: allocate the graph ONCE (via ggml_gallocr) and then only
    // re-upload inputs + compute. Re-running the scheduler's reset+alloc on every
    // replay made the result depend on the allocation pass instead of the inputs
    // (see tests/repro_graphcache.cpp) -- ggml_gallocr + alloc-once is bit-exact.
    // A graph is either sched-driven (compute) or static-driven (compute_static).
    void alloc_static();
    void compute_static();

    void compute_if_needed();
    // Force a (re)compute of the already-captured graph. Pair with set_input() to
    // replay one captured graph with fresh inputs — that is the graph-cache path
    // (the graph build is skipped; reset + alloc + compute still happen).
    void compute();
    void read(ggml_tensor* node, void* dst, size_t bytes);

    static Graph* current();

    // Make this graph the target of ops until the matching exit().
    void enter();
    void exit();

    // RAII: make this graph the target of ops for the enclosing scope.
    class Scope {
    public:
        explicit Scope(Graph* g) { g->enter(); g_ = g; }
        ~Scope() { g_->exit(); }

    private:
        Graph* g_;
    };

private:
    void init(void* meta_buf, size_t meta_bytes, size_t max_nodes);

    ggml_backend_t backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    Device device_;
    ggml_context* ctx_ = nullptr;
    ggml_cgraph* gf_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<uint8_t> owned_meta_;
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> inputs_;
    Graph* prev_ = nullptr;
    bool computed_ = false;
};

}  // namespace vt
