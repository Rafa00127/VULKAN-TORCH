// c_api.h — C ABI for minitorch (for .NET / P-Invoke and other FFI consumers).
//
// Tensors are opaque ggml_tensor* handles owned by the Graph (activations) or
// by a Memory/GgufFile (weights) — the caller never allocates tensor wrappers.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef MT_SHARED
#define MT_API __declspec(dllexport)
#else
#define MT_API
#endif
#else
#define MT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- runtime / device ----
MT_API void* mt_runtime_new(void);
MT_API void  mt_runtime_free(void* rt);
MT_API const char* mt_runtime_name(void* rt);
MT_API void* mt_runtime_gpu(void* rt);   // device handle
MT_API void* mt_runtime_cpu(void* rt);

// ---- gguf ----
MT_API void* mt_gguf_new(const char* path, void* device, size_t arena_bytes);
MT_API void  mt_gguf_free(void* g);
MT_API int   mt_gguf_has(void* g, const char* name);
MT_API int   mt_gguf_count(void* g);
MT_API const char* mt_gguf_name(void* g, int i);  // valid while g is alive
MT_API int   mt_gguf_tensor(void* g, const char* name, void** out_tensor);
MT_API int   mt_gguf_shape(void* g, const char* name, int64_t* out, int* ndim);

// ---- memory (persistent device tensors / weights) ----
MT_API void* mt_memory_new(void* device, size_t bytes);
MT_API void  mt_memory_free(void* m);
MT_API int   mt_memory_tensor(void* m, const int64_t* shape, int ndim, int dtype,
                              const void* data, size_t bytes, void** out_tensor);

// ---- graph (capture scope) ----
MT_API void* mt_graph_new(void* rt, void* device);
MT_API void  mt_graph_free(void* g);
MT_API void  mt_graph_enter(void* g);
MT_API void  mt_graph_exit(void* g);
MT_API int   mt_graph_input(void* g, const int64_t* shape, int ndim, const void* data,
                            size_t bytes, void** out_tensor);
MT_API int   mt_graph_input_i32(void* g, const int64_t* shape, int ndim, const void* data,
                                size_t bytes, void** out_tensor);
MT_API void  mt_graph_to_bytes(void* g, void* tensor, void* out, size_t bytes);

// ---- tensor metadata ----
MT_API int     mt_tensor_dim(void* t);
MT_API int64_t mt_tensor_numel(void* t);
MT_API void    mt_tensor_shape(void* t, int64_t* out);
MT_API void    mt_tensor_mark_output(void* t);
MT_API const char* mt_tensor_backend_name(void* t);

// ---- ops (PyTorch semantics; every op appends to the current Graph) ----
MT_API void* mt_matmul(void* a, void* b);
MT_API void* mt_mul_mat(void* a, void* b);
MT_API void* mt_add(void* a, void* b);
MT_API void* mt_sub(void* a, void* b);
MT_API void* mt_mul(void* a, void* b);
MT_API void* mt_scale(void* a, float s);
MT_API void* mt_transpose(void* a);
MT_API void* mt_contiguous(void* a);
MT_API void* mt_reshape(void* a, const int64_t* shape, int ndim);
MT_API void* mt_repeat(void* a, const int64_t* shape, int ndim);
MT_API void* mt_gelu(void* a);
MT_API void* mt_elu(void* a);
MT_API void* mt_silu(void* a);
MT_API void* mt_tanh(void* a);
MT_API void* mt_soft_max(void* a);
MT_API void* mt_linear(void* x, void* w);
MT_API void* mt_rms_norm(void* a, float eps);
MT_API void* mt_layer_norm(void* a, float eps);
MT_API void* mt_concat(void* a, void* b, int64_t pt_dim);
MT_API void* mt_argmax(void* a);
MT_API void* mt_get_rows(void* a, void* ids);
MT_API void* mt_sum_rows(void* a);
MT_API void* mt_cast(void* a, int type);
MT_API void* mt_cpy(void* a, void* dst);
MT_API void* mt_view_2d(void* a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
MT_API void* mt_view_3d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2,
                        size_t offset);
MT_API void* mt_view_4d(void* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1,
                        size_t nb2, size_t nb3, size_t offset);
MT_API void* mt_permute_pt(void* a, int p0, int p1, int p2, int p3);
MT_API void* mt_rope(void* a, void* pos, int n_dims, int mode, int n_ctx_orig, float freq_base,
                     float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                     float beta_slow);
MT_API void* mt_flash_attn(void* q, void* k, void* v, void* mask, float scale, float max_bias,
                           float logit_softcap);  // mask may be NULL
MT_API void* mt_conv1d(void* x, void* w, int stride, int pad, int dilation);
MT_API void* mt_conv_transpose_1d(void* x, void* w_perm, int stride, int oc);
MT_API void* mt_snake_1d(void* x, void* alpha);
MT_API void* mt_im2col_rafa(void* x, int K, int s0, int p0, int d0);
MT_API void* mt_col2im_1d(void* col, int s0, int oc, int p0);

#ifdef __cplusplus
}
#endif
