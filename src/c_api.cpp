#include "c_api.h"

#include "ggml.h"

#include "gguf_file.h"
#include "graph.h"
#include "memory.h"
#include "ops.h"
#include "runtime.h"
#include "tensor.h"

#include <cstring>
#include <new>

using namespace mt;

static Tensor wrap(void* p) { return Tensor(static_cast<ggml_tensor*>(p), Graph::current()); }
static void* unwrap(const Tensor& t) { return t.raw(); }

extern "C" {

// ---- runtime / device ----
void* mt_runtime_new(void) { return new (std::nothrow) Runtime(); }
void mt_runtime_free(void* rt) { delete static_cast<Runtime*>(rt); }
const char* mt_runtime_name(void* rt) { return rt ? static_cast<Runtime*>(rt)->name() : ""; }
void* mt_runtime_gpu(void* rt) { return new Device(static_cast<Runtime*>(rt)->gpu()); }
void* mt_runtime_cpu(void* rt) { return new Device(static_cast<Runtime*>(rt)->cpu()); }

// ---- gguf ----
void* mt_gguf_new(const char* path, void* device, size_t arena_bytes) {
    return new (std::nothrow) GgufFile(path, *static_cast<Device*>(device), arena_bytes);
}
void mt_gguf_free(void* g) { delete static_cast<GgufFile*>(g); }
int mt_gguf_has(void* g, const char* name) {
    return static_cast<GgufFile*>(g)->has(name) ? 1 : 0;
}
int mt_gguf_count(void* g) { return static_cast<GgufFile*>(g)->count(); }
const char* mt_gguf_name(void* g, int i) { return static_cast<GgufFile*>(g)->name_at(i); }
int mt_gguf_tensor(void* g, const char* name, void** out_tensor) {
    try {
        *out_tensor = unwrap(static_cast<GgufFile*>(g)->tensor(name));
        return 0;
    } catch (...) {
        return -1;
    }
}
int mt_gguf_shape(void* g, const char* name, int64_t* out, int* ndim) {
    try {
        auto s = static_cast<GgufFile*>(g)->shape(name);
        for (size_t i = 0; i < s.size(); ++i) out[i] = s[i];
        *ndim = static_cast<int>(s.size());
        return 0;
    } catch (...) {
        return -1;
    }
}

// ---- memory ----
void* mt_memory_new(void* device, size_t bytes) {
    return new (std::nothrow) Memory(*static_cast<Device*>(device), bytes);
}
void mt_memory_free(void* m) { delete static_cast<Memory*>(m); }
int mt_memory_tensor(void* m, const int64_t* shape, int ndim, int dtype, const void* data,
                     size_t bytes, void** out_tensor) {
    try {
        std::vector<int64_t> sh(shape, shape + ndim);
        *out_tensor = unwrap(static_cast<Memory*>(m)->tensor(sh, static_cast<ggml_type>(dtype), data,
                                                             bytes));
        return 0;
    } catch (...) {
        return -1;
    }
}

// ---- graph ----
void* mt_graph_new(void* rt, void* device) {
    return new (std::nothrow) Graph(*static_cast<Runtime*>(rt), *static_cast<Device*>(device));
}
void mt_graph_free(void* g) { delete static_cast<Graph*>(g); }
void mt_graph_enter(void* g) { static_cast<Graph*>(g)->enter(); }
void mt_graph_exit(void* g) { static_cast<Graph*>(g)->exit(); }
int mt_graph_input(void* g, const int64_t* shape, int ndim, const void* data, size_t bytes,
                   void** out_tensor) {
    try {
        std::vector<int64_t> sh(shape, shape + ndim);
        *out_tensor = unwrap(static_cast<Graph*>(g)->input(sh, data, bytes));
        return 0;
    } catch (...) {
        return -1;
    }
}
int mt_graph_input_i32(void* g, const int64_t* shape, int ndim, const void* data, size_t bytes,
                       void** out_tensor) {
    try {
        std::vector<int64_t> sh(shape, shape + ndim);
        *out_tensor = unwrap(static_cast<Graph*>(g)->input_i32(sh, data, bytes));
        return 0;
    } catch (...) {
        return -1;
    }
}
void mt_graph_to_bytes(void* g, void* tensor, void* out, size_t bytes) {
    std::string s = Tensor(static_cast<ggml_tensor*>(tensor), static_cast<Graph*>(g)).to_host_bytes();
    std::memcpy(out, s.data(), bytes < s.size() ? bytes : s.size());
}

// ---- tensor metadata ----
int mt_tensor_dim(void* t) { return ggml_n_dims(static_cast<ggml_tensor*>(t)); }
int64_t mt_tensor_numel(void* t) { return ggml_nelements(static_cast<ggml_tensor*>(t)); }
void mt_tensor_shape(void* t, int64_t* out) {
    ggml_tensor* tt = static_cast<ggml_tensor*>(t);
    int nd = ggml_n_dims(tt);
    for (int i = 0; i < nd; ++i) out[i] = tt->ne[nd - 1 - i];
}
void mt_tensor_mark_output(void* t) {
    if (t) ggml_set_output(static_cast<ggml_tensor*>(t));
}
const char* mt_tensor_backend_name(void* t) {
    ggml_tensor* tt = static_cast<ggml_tensor*>(t);
    return (tt && tt->buffer) ? ggml_backend_buffer_name(tt->buffer) : "";
}

// ---- ops ----
void* mt_matmul(void* a, void* b) { return unwrap(matmul(wrap(a), wrap(b))); }
void* mt_mul_mat(void* a, void* b) { return unwrap(mul_mat(wrap(a), wrap(b))); }
void* mt_add(void* a, void* b) { return unwrap(add(wrap(a), wrap(b))); }
void* mt_sub(void* a, void* b) { return unwrap(sub(wrap(a), wrap(b))); }
void* mt_mul(void* a, void* b) { return unwrap(mul(wrap(a), wrap(b))); }
void* mt_scale(void* a, float s) { return unwrap(scale(wrap(a), s)); }
void* mt_transpose(void* a) { return unwrap(transpose(wrap(a))); }
void* mt_contiguous(void* a) { return unwrap(contiguous(wrap(a))); }
void* mt_reshape(void* a, const int64_t* shape, int ndim) {
    return unwrap(reshape(wrap(a), std::vector<int64_t>(shape, shape + ndim)));
}
void* mt_repeat(void* a, const int64_t* shape, int ndim) {
    return unwrap(repeat(wrap(a), std::vector<int64_t>(shape, shape + ndim)));
}
void* mt_gelu(void* a) { return unwrap(gelu(wrap(a))); }
void* mt_elu(void* a) { return unwrap(elu(wrap(a))); }
void* mt_silu(void* a) { return unwrap(silu(wrap(a))); }
void* mt_tanh(void* a) { return unwrap(tanh(wrap(a))); }
void* mt_soft_max(void* a) { return unwrap(soft_max(wrap(a))); }
void* mt_linear(void* x, void* w) { return unwrap(linear(wrap(x), wrap(w))); }
void* mt_rms_norm(void* a, float eps) { return unwrap(rms_norm(wrap(a), eps)); }
void* mt_layer_norm(void* a, float eps) { return unwrap(layer_norm(wrap(a), eps)); }
void* mt_concat(void* a, void* b, int64_t d) { return unwrap(concat(wrap(a), wrap(b), d)); }
void* mt_argmax(void* a) { return unwrap(argmax(wrap(a))); }
void* mt_get_rows(void* a, void* ids) { return unwrap(get_rows(wrap(a), wrap(ids))); }
void* mt_sum_rows(void* a) { return unwrap(sum_rows(wrap(a))); }
void* mt_cast(void* a, int type) { return unwrap(cast(wrap(a), static_cast<ggml_type>(type))); }
void* mt_cpy(void* a, void* dst) { return unwrap(cpy(wrap(a), wrap(dst))); }
void* mt_view_2d(void* a, int64_t ne0, int64_t ne1, size_t nb1, size_t off) {
    return unwrap(view_2d(wrap(a), ne0, ne1, nb1, off));
}
void* mt_view_3d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t off) {
    return unwrap(view_3d(wrap(a), ne0, ne1, ne2, nb1, nb2, off));
}
void* mt_view_4d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2,
                 size_t nb3, size_t off) {
    return unwrap(view_4d(wrap(a), ne0, ne1, ne2, ne3, nb1, nb2, nb3, off));
}
void* mt_permute_pt(void* a, int p0, int p1, int p2, int p3) {
    return unwrap(permute_pt(wrap(a), p0, p1, p2, p3));
}
void* mt_rope(void* a, void* pos, int n_dims, int mode, int n_ctx_orig, float fb, float fs,
              float ef, float af, float bf, float bs) {
    Tensor p = pos ? wrap(pos) : Tensor();
    return unwrap(rope(wrap(a), p, n_dims, mode, n_ctx_orig, fb, fs, ef, af, bf, bs));
}
void* mt_flash_attn(void* q, void* k, void* v, void* mask, float scale, float mb, float ls) {
    Tensor m = mask ? wrap(mask) : Tensor();
    return unwrap(flash_attn(wrap(q), wrap(k), wrap(v), m, scale, mb, ls));
}
void* mt_conv1d(void* x, void* w, int stride, int pad, int dil) {
    return unwrap(conv1d(wrap(x), wrap(w), stride, pad, dil));
}
void* mt_conv_transpose_1d(void* x, void* wp, int stride, int oc) {
    return unwrap(conv_transpose_1d(wrap(x), wrap(wp), stride, oc));
}
void* mt_snake_1d(void* x, void* alpha) { return unwrap(snake_1d(wrap(x), wrap(alpha))); }
void* mt_im2col_rafa(void* x, int K, int s0, int p0, int d0) {
    return unwrap(im2col_rafa(wrap(x), K, s0, p0, d0, GGML_TYPE_F32));
}
void* mt_col2im_1d(void* col, int s0, int oc, int p0) {
    return unwrap(col2im_1d(wrap(col), s0, oc, p0));
}

}  // extern "C"
