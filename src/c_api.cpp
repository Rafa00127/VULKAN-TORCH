#include "c_api.h"

#include "ggml.h"

#include "gguf_file.h"
#include "graph.h"
#include "memory.h"
#include "ops.h"
#include "runtime.h"
#include "tensor.h"

#include <cstring>
#include <exception>
#include <string>

using namespace vt;

static Tensor wrap(void* p) { return Tensor(static_cast<ggml_tensor*>(p), Graph::current()); }
static void* unwrap(const Tensor& t) { return t.raw(); }

// ---- error channel ----------------------------------------------------------
//
// A C++ exception must not cross the C ABI: on the .NET side that is undefined
// behaviour and the process dies with no output at all (no managed exception, no
// stack, exit 127). So every fallible entry point below catches, stashes the
// message here, and returns a sentinel; the managed side reads the message back
// through vt_last_error() and raises an ordinary exception. The intended default
// is still death -- an uncaught managed exception -- just an informative one.
//
// thread_local, like Graph::current(): a P/Invoke call is synchronous on the
// calling thread, so the producer and the consumer are always the same thread,
// and concurrent callers cannot clobber each other's message.
static thread_local std::string t_error;

// Clear on entry, so a stale message is never mistaken for a fresh failure.
struct ErrGuard {
    ErrGuard() { t_error.clear(); }
};

// Wrap a fallible body that yields the function's return value. `body` must be a
// single expression with no top-level comma.
#define VT_TRY(body, sentinel)                                                  \
    ErrGuard vt_err_guard_;                                                     \
    try {                                                                       \
        return (body);                                                          \
    } catch (const std::exception& e) {                                         \
        t_error = e.what();                                                     \
        return (sentinel);                                                      \
    } catch (...) {                                                             \
        t_error = "unknown C++ exception";                                      \
        return (sentinel);                                                      \
    }

// Same, for bodies that are statements rather than one expression -- a braced
// block, possibly with its own `return` for the value, and/or a separate success
// code returned by the caller (the `int`-returning entry points).
#define VT_TRY_VOID(body, sentinel)                                             \
    ErrGuard vt_err_guard_;                                                     \
    try {                                                                       \
        body;                                                                   \
    } catch (const std::exception& e) {                                         \
        t_error = e.what();                                                     \
        return (sentinel);                                                      \
    } catch (...) {                                                             \
        t_error = "unknown C++ exception";                                      \
        return (sentinel);                                                      \
    }

// GGML_ASSERT still reaches abort() and cannot be intercepted -- it goes through
// ggml_abort(), which is unconditional. It is also not silent: ggml's default
// handler already prints the file, line and assertion to stderr before dying. So
// no abort callback is installed; the message stays on stderr, as it always was.

extern "C" {

// Message from the most recent failed vt_* call on this thread; "" if none.
const char* vt_last_error(void) { return t_error.c_str(); }

// ---- runtime / device ----
void* vt_runtime_new(void) { VT_TRY(new Runtime(), nullptr); }
void vt_runtime_free(void* rt) { delete static_cast<Runtime*>(rt); }
const char* vt_runtime_name(void* rt) { return rt ? static_cast<Runtime*>(rt)->name() : ""; }
void* vt_runtime_gpu(void* rt) {
    VT_TRY(new Device(static_cast<Runtime*>(rt)->gpu()), nullptr);
}
void* vt_runtime_cpu(void* rt) {
    VT_TRY(new Device(static_cast<Runtime*>(rt)->cpu()), nullptr);
}

// ---- gguf ----
void* vt_gguf_new(const char* path, void* device, size_t arena_bytes) {
    VT_TRY(new GgufFile(path, *static_cast<Device*>(device), arena_bytes), nullptr);
}
void vt_gguf_free(void* g) { delete static_cast<GgufFile*>(g); }
int vt_gguf_has(void* g, const char* name) {
    return static_cast<GgufFile*>(g)->has(name) ? 1 : 0;
}
int vt_gguf_count(void* g) { return static_cast<GgufFile*>(g)->count(); }
const char* vt_gguf_name(void* g, int i) { return static_cast<GgufFile*>(g)->name_at(i); }
int vt_gguf_tensor(void* g, const char* name, void** out_tensor) {
    VT_TRY_VOID(
        {
            *out_tensor = unwrap(static_cast<GgufFile*>(g)->tensor(name));
            return 0;
        },
        -1);
}
int vt_gguf_shape(void* g, const char* name, int64_t* out, int* ndim) {
    VT_TRY_VOID(
        {
            auto s = static_cast<GgufFile*>(g)->shape(name);
            for (size_t i = 0; i < s.size(); ++i) out[i] = s[i];
            *ndim = static_cast<int>(s.size());
            return 0;
        },
        -1);
}

// ---- memory ----
void* vt_memory_new(void* device, size_t bytes) {
    VT_TRY(new Memory(*static_cast<Device*>(device), bytes), nullptr);
}
void vt_memory_free(void* m) { delete static_cast<Memory*>(m); }
int vt_memory_tensor(void* m, const int64_t* shape, int ndim, int dtype, const void* data,
                     size_t bytes, void** out_tensor) {
    VT_TRY_VOID(
        {
            std::vector<int64_t> sh(shape, shape + ndim);
            *out_tensor = unwrap(static_cast<Memory*>(m)->tensor(
                sh, static_cast<ggml_type>(dtype), data, bytes));
            return 0;
        },
        -1);
}

// ---- graph ----
void* vt_graph_new(void* rt, void* device) {
    VT_TRY(new Graph(*static_cast<Runtime*>(rt), *static_cast<Device*>(device)), nullptr);
}
void* vt_graph_new_n(void* rt, void* device, size_t max_nodes) {
    VT_TRY(new Graph(*static_cast<Runtime*>(rt), *static_cast<Device*>(device), max_nodes),
           nullptr);
}
void vt_graph_free(void* g) { delete static_cast<Graph*>(g); }
void vt_graph_enter(void* g) { static_cast<Graph*>(g)->enter(); }
void vt_graph_exit(void* g) { static_cast<Graph*>(g)->exit(); }
int vt_graph_input(void* g, const int64_t* shape, int ndim, const void* data, size_t bytes,
                   void** out_tensor) {
    VT_TRY_VOID(
        {
            std::vector<int64_t> sh(shape, shape + ndim);
            *out_tensor = unwrap(static_cast<Graph*>(g)->input(sh, data, bytes));
            return 0;
        },
        -1);
}
int vt_graph_input_i32(void* g, const int64_t* shape, int ndim, const void* data, size_t bytes,
                       void** out_tensor) {
    VT_TRY_VOID(
        {
            std::vector<int64_t> sh(shape, shape + ndim);
            *out_tensor = unwrap(static_cast<Graph*>(g)->input_i32(sh, data, bytes));
            return 0;
        },
        -1);
}
int vt_graph_to_bytes(void* g, void* tensor, void* out, size_t bytes) {
    VT_TRY_VOID(
        {
            std::string s =
                Tensor(static_cast<ggml_tensor*>(tensor), static_cast<Graph*>(g)).to_host_bytes();
            std::memcpy(out, s.data(), bytes < s.size() ? bytes : s.size());
            return 0;
        },
        0);
}
int vt_graph_n_nodes(void* g) { return ggml_graph_n_nodes(static_cast<Graph*>(g)->cgraph()); }
int vt_graph_compute(void* g) {
    VT_TRY_VOID(static_cast<Graph*>(g)->compute(), -1);
    return 0;
}
int vt_graph_alloc_static(void* g) {
    VT_TRY_VOID(static_cast<Graph*>(g)->alloc_static(), -1);
    return 0;
}
int vt_graph_compute_static(void* g) {
    VT_TRY_VOID(static_cast<Graph*>(g)->compute_static(), -1);
    return 0;
}
int vt_graph_set_input(void* g, void* tensor, const void* data, size_t bytes) {
    VT_TRY_VOID(
        {
            static_cast<Graph*>(g)->set_input(static_cast<ggml_tensor*>(tensor), data, bytes);
            return 0;
        },
        -1);
}

// ---- tensor metadata ----
int vt_tensor_dim(void* t) { return ggml_n_dims(static_cast<ggml_tensor*>(t)); }
int64_t vt_tensor_numel(void* t) { return ggml_nelements(static_cast<ggml_tensor*>(t)); }
void vt_tensor_shape(void* t, int64_t* out) {
    ggml_tensor* tt = static_cast<ggml_tensor*>(t);
    int nd = ggml_n_dims(tt);
    for (int i = 0; i < nd; ++i) out[i] = tt->ne[nd - 1 - i];
}
void vt_tensor_mark_output(void* t) {
    if (t) ggml_set_output(static_cast<ggml_tensor*>(t));
}
const char* vt_tensor_backend_name(void* t) {
    ggml_tensor* tt = static_cast<ggml_tensor*>(t);
    return (tt && tt->buffer) ? ggml_backend_buffer_name(tt->buffer) : "";
}
int vt_tensor_type(void* t) { return static_cast<int>(static_cast<ggml_tensor*>(t)->type); }
void* vt_tensor_data_ptr(void* t) { return t ? static_cast<ggml_tensor*>(t)->data : nullptr; }
size_t vt_tensor_nbytes(void* t) { return ggml_nbytes(static_cast<ggml_tensor*>(t)); }
int vt_tensor_to_bytes(void* t, void* out, size_t bytes) {
    VT_TRY_VOID(Tensor(static_cast<ggml_tensor*>(t), nullptr).to_host_bytes_into(out, bytes), -1);
    return 0;
}

// ---- ops ----
void* vt_matmul(void* a, void* b) { VT_TRY(unwrap(matmul(wrap(a), wrap(b))), nullptr); }
void* vt_mul_mat(void* a, void* b) { VT_TRY(unwrap(mul_mat(wrap(a), wrap(b))), nullptr); }
void* vt_add(void* a, void* b) { VT_TRY(unwrap(add(wrap(a), wrap(b))), nullptr); }
void* vt_sub(void* a, void* b) { VT_TRY(unwrap(sub(wrap(a), wrap(b))), nullptr); }
void* vt_mul(void* a, void* b) { VT_TRY(unwrap(mul(wrap(a), wrap(b))), nullptr); }
void* vt_div(void* a, void* b) { VT_TRY(unwrap(div(wrap(a), wrap(b))), nullptr); }
void* vt_scale(void* a, float s) { VT_TRY(unwrap(scale(wrap(a), s)), nullptr); }
void* vt_scale_bias(void* a, float s, float b) { VT_TRY(unwrap(scale_bias(wrap(a), s, b)), nullptr); }
void* vt_transpose(void* a) { VT_TRY(unwrap(transpose(wrap(a))), nullptr); }
void* vt_contiguous(void* a) { VT_TRY(unwrap(contiguous(wrap(a))), nullptr); }
void* vt_reshape(void* a, const int64_t* shape, int ndim) {
    VT_TRY(unwrap(reshape(wrap(a), std::vector<int64_t>(shape, shape + ndim))), nullptr);
}
void* vt_repeat(void* a, const int64_t* shape, int ndim) {
    VT_TRY(unwrap(repeat(wrap(a), std::vector<int64_t>(shape, shape + ndim))), nullptr);
}
void* vt_gelu(void* a) { VT_TRY(unwrap(gelu(wrap(a))), nullptr); }
void* vt_relu(void* a) { VT_TRY(unwrap(relu(wrap(a))), nullptr); }
void* vt_sigmoid(void* a) { VT_TRY(unwrap(sigmoid(wrap(a))), nullptr); }
void* vt_exp(void* a) { VT_TRY(unwrap(exp(wrap(a))), nullptr); }
void* vt_elu(void* a) { VT_TRY(unwrap(elu(wrap(a))), nullptr); }
void* vt_silu(void* a) { VT_TRY(unwrap(silu(wrap(a))), nullptr); }
void* vt_tanh(void* a) { VT_TRY(unwrap(tanh(wrap(a))), nullptr); }
void* vt_soft_max(void* a) { VT_TRY(unwrap(soft_max(wrap(a))), nullptr); }
void* vt_softplus(void* a) { VT_TRY(unwrap(softplus(wrap(a))), nullptr); }
void* vt_sin(void* a) { VT_TRY(unwrap(sin(wrap(a))), nullptr); }
void* vt_cos(void* a) { VT_TRY(unwrap(cos(wrap(a))), nullptr); }
void* vt_sqrt(void* a) { VT_TRY(unwrap(sqrt(wrap(a))), nullptr); }
void* vt_sqr(void* a) { VT_TRY(unwrap(sqr(wrap(a))), nullptr); }
void* vt_linear(void* x, void* w) { VT_TRY(unwrap(linear(wrap(x), wrap(w))), nullptr); }
void* vt_rms_norm(void* a, float eps) { VT_TRY(unwrap(rms_norm(wrap(a), eps)), nullptr); }
void* vt_layer_norm(void* a, float eps) { VT_TRY(unwrap(layer_norm(wrap(a), eps)), nullptr); }
void* vt_group_norm(void* a, int n_groups, float eps) {
    VT_TRY(unwrap(group_norm(wrap(a), n_groups, eps)), nullptr);
}
void* vt_diag_mask_inf(void* a, int n_past) {
    VT_TRY(unwrap(diag_mask_inf(wrap(a), n_past)), nullptr);
}
void* vt_concat(void* a, void* b, int64_t d) { VT_TRY(unwrap(concat(wrap(a), wrap(b), d)), nullptr); }
void* vt_argmax(void* a) { VT_TRY(unwrap(argmax(wrap(a))), nullptr); }
void* vt_get_rows(void* a, void* ids) { VT_TRY(unwrap(get_rows(wrap(a), wrap(ids))), nullptr); }
void* vt_sum_rows(void* a) { VT_TRY(unwrap(sum_rows(wrap(a))), nullptr); }
void* vt_cast(void* a, int type) {
    VT_TRY(unwrap(cast(wrap(a), static_cast<ggml_type>(type))), nullptr);
}
void* vt_cpy(void* a, void* dst) { VT_TRY(unwrap(cpy(wrap(a), wrap(dst))), nullptr); }
void* vt_set_rows(void* dst, void* src, void* idx) {
    VT_TRY(unwrap(set_rows(wrap(dst), wrap(src), wrap(idx))), nullptr);
}
void* vt_view_1d(void* a, int64_t ne0, size_t off) {
    VT_TRY(unwrap(view_1d(wrap(a), ne0, off)), nullptr);
}
void* vt_view_2d(void* a, int64_t ne0, int64_t ne1, size_t nb1, size_t off) {
    VT_TRY(unwrap(view_2d(wrap(a), ne0, ne1, nb1, off)), nullptr);
}
void* vt_view_3d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t off) {
    VT_TRY(unwrap(view_3d(wrap(a), ne0, ne1, ne2, nb1, nb2, off)), nullptr);
}
void* vt_view_4d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2,
                 size_t nb3, size_t off) {
    VT_TRY(unwrap(view_4d(wrap(a), ne0, ne1, ne2, ne3, nb1, nb2, nb3, off)), nullptr);
}
void* vt_permute_pt(void* a, int p0, int p1, int p2, int p3) {
    VT_TRY(unwrap(permute_pt(wrap(a), p0, p1, p2, p3)), nullptr);
}
void* vt_rope(void* a, void* pos, int n_dims, int mode, int n_ctx_orig, float fb, float fs,
              float ef, float af, float bf, float bs) {
    VT_TRY_VOID(
        {
            Tensor p = pos != nullptr ? wrap(pos) : Tensor();
            return unwrap(rope(wrap(a), p, n_dims, mode, n_ctx_orig, fb, fs, ef, af, bf, bs));
        },
        nullptr);
}
void* vt_rope_multi(void* a, void* pos, int n_dims, int s0, int s1, int s2, int s3, int mode,
                    int n_ctx_orig, float fb, float fs, float ef, float af, float bf, float bs) {
    // built outside VT_TRY_VOID: a braced initialiser's commas would split the macro's
    // arguments (the preprocessor groups on parentheses only).
    int sec[4];
    sec[0] = s0;
    sec[1] = s1;
    sec[2] = s2;
    sec[3] = s3;
    VT_TRY_VOID(
        {
            Tensor p = pos != nullptr ? wrap(pos) : Tensor();
            return unwrap(rope_multi(wrap(a), p, n_dims, sec, mode, n_ctx_orig, fb, fs, ef, af, bf,
                                     bs));
        },
        nullptr);
}
void* vt_flash_attn(void* q, void* k, void* v, void* mask, float scale, float mb, float ls) {
    VT_TRY_VOID(
        {
            Tensor m = mask != nullptr ? wrap(mask) : Tensor();
            return unwrap(flash_attn(wrap(q), wrap(k), wrap(v), m, scale, mb, ls));
        },
        nullptr);
}
void* vt_conv1d(void* x, void* w, int stride, int pad, int dil) {
    VT_TRY(unwrap(conv1d(wrap(x), wrap(w), stride, pad, dil)), nullptr);
}
void* vt_conv2d(void* a, void* b, int s0, int s1, int p0, int p1, int d0, int d1) {
    VT_TRY(unwrap(conv2d(wrap(a), wrap(b), s0, s1, p0, p1, d0, d1)), nullptr);
}
void* vt_conv2d_tiled(void* a, void* b, int s0, int s1, int p0, int p1, int d0, int d1,
                      int n_tiles) {
    VT_TRY(unwrap(conv2d_tiled(wrap(a), wrap(b), s0, s1, p0, p1, d0, d1, n_tiles)), nullptr);
}
void* vt_conv1d_dw(void* x, void* w, int stride, int pad, int dil) {
    VT_TRY(unwrap(conv1d_dw(wrap(x), wrap(w), stride, pad, dil)), nullptr);
}
void* vt_conv_transpose_1d(void* x, void* wp, int stride, int oc) {
    VT_TRY(unwrap(conv_transpose_1d(wrap(x), wrap(wp), stride, oc)), nullptr);
}
void* vt_snake_1d(void* x, void* alpha) { VT_TRY(unwrap(snake_1d(wrap(x), wrap(alpha))), nullptr); }
void* vt_im2col_rafa(void* x, int K, int s0, int p0, int d0) {
    VT_TRY(unwrap(im2col_rafa(wrap(x), K, s0, p0, d0, GGML_TYPE_F32)), nullptr);
}
void* vt_col2im_1d(void* col, int s0, int oc, int p0) {
    VT_TRY(unwrap(col2im_1d(wrap(col), s0, oc, p0)), nullptr);
}
void* vt_conv2d_dw(void* a, void* b, int s0, int s1, int p0, int p1, int d0, int d1) {
    VT_TRY(unwrap(conv2d_dw(wrap(a), wrap(b), s0, s1, p0, p1, d0, d1)), nullptr);
}
void* vt_conv_transpose_2d(void* a, void* b, int stride) {
    VT_TRY(unwrap(conv_transpose_2d(wrap(a), wrap(b), stride)), nullptr);
}
void* vt_pool_2d(void* a, int op, int k0, int k1, int s0, int s1, float p0, float p1) {
    VT_TRY(unwrap(pool_2d(wrap(a), op, k0, k1, s0, s1, p0, p1)), nullptr);
}
void* vt_upsample(void* a, int scale_factor, int mode) {
    VT_TRY(unwrap(upsample(wrap(a), scale_factor, mode)), nullptr);
}
void* vt_pad(void* a, int p0, int p1, int p2, int p3) {
    VT_TRY(unwrap(pad(wrap(a), p0, p1, p2, p3)), nullptr);
}
void* vt_clamp(void* a, float min_v, float max_v) {
    VT_TRY(unwrap(clamp(wrap(a), min_v, max_v)), nullptr);
}
void* vt_gelu_erf(void* a) { VT_TRY(unwrap(gelu_erf(wrap(a))), nullptr); }

}  // extern "C"
