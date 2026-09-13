// c_api.h — C ABI for vulkantorch (for .NET / P-Invoke and other FFI consumers).
//
// Tensors are opaque ggml_tensor* handles owned by the Graph (activations) or
// by a Memory/GgufFile (weights) — the caller never allocates tensor wrappers.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef VT_SHARED
#define VT_API __declspec(dllexport)
#else
#define VT_API
#endif
#else
#define VT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- runtime / device ----
VT_API void* vt_runtime_new(void);
VT_API void  vt_runtime_free(void* rt);
VT_API const char* vt_runtime_name(void* rt);
VT_API void* vt_runtime_gpu(void* rt);   // device handle
VT_API void* vt_runtime_cpu(void* rt);

// ---- gguf ----
VT_API void* vt_gguf_new(const char* path, void* device, size_t arena_bytes);
VT_API void  vt_gguf_free(void* g);
VT_API int   vt_gguf_has(void* g, const char* name);
VT_API int   vt_gguf_count(void* g);
VT_API const char* vt_gguf_name(void* g, int i);  // valid while g is alive
VT_API int   vt_gguf_tensor(void* g, const char* name, void** out_tensor);
VT_API int   vt_gguf_shape(void* g, const char* name, int64_t* out, int* ndim);

// ---- memory (persistent device tensors / weights) ----
VT_API void* vt_memory_new(void* device, size_t bytes);
VT_API void  vt_memory_free(void* m);
VT_API int   vt_memory_tensor(void* m, const int64_t* shape, int ndim, int dtype,
                              const void* data, size_t bytes, void** out_tensor);

// ---- graph (capture scope) ----
VT_API void* vt_graph_new(void* rt, void* device);
VT_API void* vt_graph_new_n(void* rt, void* device, size_t max_nodes);
VT_API void  vt_graph_free(void* g);
VT_API void  vt_graph_enter(void* g);
VT_API void  vt_graph_exit(void* g);
VT_API int   vt_graph_input(void* g, const int64_t* shape, int ndim, const void* data,
                            size_t bytes, void** out_tensor);
VT_API int   vt_graph_input_i32(void* g, const int64_t* shape, int ndim, const void* data,
                                size_t bytes, void** out_tensor);
VT_API void  vt_graph_to_bytes(void* g, void* tensor, void* out, size_t bytes);
// Graph-cache support: replay one captured graph with fresh inputs.
VT_API int   vt_graph_n_nodes(void* g);            // captured-graph size (introspection)
VT_API void  vt_graph_compute(void* g);
VT_API void  vt_graph_alloc_static(void* g);
VT_API void  vt_graph_compute_static(void* g);
VT_API int   vt_graph_set_input(void* g, void* tensor, const void* data, size_t bytes);

// ---- tensor metadata ----
VT_API int     vt_tensor_dim(void* t);
VT_API int64_t vt_tensor_numel(void* t);
VT_API void    vt_tensor_shape(void* t, int64_t* out);
VT_API void    vt_tensor_mark_output(void* t);
VT_API const char* vt_tensor_backend_name(void* t);
VT_API int     vt_tensor_type(void* t);            // ggml_type
VT_API void*   vt_tensor_data_ptr(void* t);        // device buffer address (introspection)
VT_API size_t  vt_tensor_nbytes(void* t);
VT_API void    vt_tensor_to_bytes(void* t, void* out, size_t bytes);  // raw device readback

// ---- ops (PyTorch semantics; every op appends to the current Graph) ----
VT_API void* vt_matmul(void* a, void* b);
VT_API void* vt_mul_mat(void* a, void* b);
VT_API void* vt_add(void* a, void* b);
VT_API void* vt_sub(void* a, void* b);
VT_API void* vt_mul(void* a, void* b);
VT_API void* vt_div(void* a, void* b);
VT_API void* vt_scale(void* a, float s);
VT_API void* vt_scale_bias(void* a, float s, float b);
VT_API void* vt_transpose(void* a);
VT_API void* vt_contiguous(void* a);
VT_API void* vt_reshape(void* a, const int64_t* shape, int ndim);
VT_API void* vt_repeat(void* a, const int64_t* shape, int ndim);
VT_API void* vt_gelu(void* a);
VT_API void* vt_relu(void* a);
VT_API void* vt_sigmoid(void* a);
VT_API void* vt_exp(void* a);
VT_API void* vt_elu(void* a);
VT_API void* vt_silu(void* a);
VT_API void* vt_tanh(void* a);
VT_API void* vt_soft_max(void* a);
VT_API void* vt_softplus(void* a);
VT_API void* vt_sin(void* a);
VT_API void* vt_cos(void* a);
VT_API void* vt_sqrt(void* a);
VT_API void* vt_sqr(void* a);
VT_API void* vt_linear(void* x, void* w);
VT_API void* vt_rms_norm(void* a, float eps);
VT_API void* vt_layer_norm(void* a, float eps);
VT_API void* vt_group_norm(void* a, int n_groups, float eps);
VT_API void* vt_diag_mask_inf(void* a, int n_past);
VT_API void* vt_concat(void* a, void* b, int64_t pt_dim);
VT_API void* vt_argmax(void* a);
VT_API void* vt_get_rows(void* a, void* ids);
VT_API void* vt_sum_rows(void* a);
VT_API void* vt_cast(void* a, int type);
VT_API void* vt_cpy(void* a, void* dst);
VT_API void* vt_set_rows(void* dst, void* src, void* idx);
VT_API void* vt_view_1d(void* a, int64_t ne0, size_t offset);
VT_API void* vt_view_2d(void* a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
VT_API void* vt_view_3d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2,
                        size_t offset);
VT_API void* vt_view_4d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1,
                        size_t nb2, size_t nb3, size_t offset);
VT_API void* vt_permute_pt(void* a, int p0, int p1, int p2, int p3);
VT_API void* vt_rope(void* a, void* pos, int n_dims, int mode, int n_ctx_orig, float freq_base,
                     float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                     float beta_slow);
VT_API void* vt_flash_attn(void* q, void* k, void* v, void* mask, float scale, float max_bias,
                           float logit_softcap);  // mask may be NULL
VT_API void* vt_conv1d(void* x, void* w, int stride, int pad, int dilation);
VT_API void* vt_conv1d_dw(void* x, void* w, int stride, int pad, int dilation);
VT_API void* vt_conv2d(void* a, void* b, int s0, int s1, int p0, int p1, int d0, int d1);
VT_API void* vt_conv2d_tiled(void* a, void* b, int s0, int s1, int p0, int p1, int d0, int d1, int n_tiles);
VT_API void* vt_conv_transpose_1d(void* x, void* w_perm, int stride, int oc);
VT_API void* vt_snake_1d(void* x, void* alpha);
VT_API void* vt_im2col_rafa(void* x, int K, int s0, int p0, int d0);
VT_API void* vt_col2im_1d(void* col, int s0, int oc, int p0);
VT_API void* vt_conv2d_dw(void* a, void* b, int s0, int s1, int p0, int p1, int d0, int d1);
VT_API void* vt_conv_transpose_2d(void* a, void* b, int stride);
VT_API void* vt_pool_2d(void* a, int op, int k0, int k1, int s0, int s1, float p0, float p1);
VT_API void* vt_upsample(void* a, int scale_factor, int mode);
VT_API void* vt_pad(void* a, int p0, int p1, int p2, int p3);
VT_API void* vt_clamp(void* a, float min_v, float max_v);
VT_API void* vt_gelu_erf(void* a);

#ifdef __cplusplus
}
#endif
