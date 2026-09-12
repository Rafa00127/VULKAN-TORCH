#pragma once

#include "tensor.h"

#include <vector>

namespace mt {

// All ops append nodes to the current Graph (see Graph::Scope) and return a
// handle. Nothing is computed until a Tensor is materialized (to_host).

Tensor matmul(const Tensor& a, const Tensor& b);  // a @ b, PyTorch semantics
Tensor mul_mat(const Tensor& a, const Tensor& b);  // raw ggml_mul_mat(a, b)

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float s);

// View / shape ops
Tensor transpose(const Tensor& a);                                   // swap last two dims
Tensor contiguous(const Tensor& a);                                  // materialize a dense copy
Tensor reshape(const Tensor& a, const std::vector<int64_t>& shape);  // PyTorch-order shape
Tensor repeat(const Tensor& a, const std::vector<int64_t>& shape);

// Activations
Tensor gelu(const Tensor& a);
Tensor elu(const Tensor& a);
Tensor silu(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor soft_max(const Tensor& a);

// Linear / norms
Tensor linear(const Tensor& x, const Tensor& w);  // x @ w^T, w is [out, in]
Tensor rms_norm(const Tensor& a, float eps);
Tensor layer_norm(const Tensor& a, float eps);

// Reductions / indexing / mixing
Tensor concat(const Tensor& a, const Tensor& b, int64_t pt_dim);
Tensor argmax(const Tensor& a);                    // along last dim -> I32 [.., 1]
Tensor get_rows(const Tensor& a, const Tensor& ids);
Tensor sum_rows(const Tensor& a);                  // sum along last dim
Tensor cast(const Tensor& a, ggml_type type);
Tensor cpy(const Tensor& a, const Tensor& dst);

// Raw view / permute (ggml ne-space; escape hatch for layout-heavy code).
Tensor view_2d(const Tensor& a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
Tensor view_3d(const Tensor& a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2,
               size_t offset);
Tensor view_4d(const Tensor& a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1,
               size_t nb2, size_t nb3, size_t offset);
Tensor permute_pt(const Tensor& a, int pt_axis0, int pt_axis1, int pt_axis2, int pt_axis3);

// Attention / rope
Tensor rope(const Tensor& a, const Tensor& pos, int n_dims, int mode, int n_ctx_orig,
            float freq_base, float freq_scale, float ext_factor, float attn_factor,
            float beta_fast, float beta_slow);
// q/k/v PT [nh, T, hd] (ggml ne=[hd,T,nh]); result PT [T, nh, hd] (ne=[hd,nh,T]).
// Input and output dim order differ -- matches ggml_flash_attn_ext. mask may be
// undefined.
Tensor flash_attn(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask,
                  float scale, float max_bias, float logit_softcap);

// Custom ops from this ggml fork (conv / activations)
// conv1d: im2col_rafa + reshape + matmul. Chosen over ggml_conv_1d (that path is
//   ~4x slower and uses a different layout). x:[T, Cin], w:[Cout, Cin, K] -> [T_out, Cout].
Tensor conv1d(const Tensor& x, const Tensor& w, int stride, int pad, int dilation);
// conv_transpose_1d via w_perm + matmul + col2im_1d. w_perm is the pre-permuted
//   weight [K*OC, IC] (see convert_model: t.reshape(IC, K*OC).T). x:[T, Cin] ->
//   PT [OC, T_out]. The hand-written ggml_conv_transpose_1d op is numerically
//   wrong here and is not exposed.
Tensor conv_transpose_1d(const Tensor& x, const Tensor& w_perm, int stride, int oc);
Tensor im2col_rafa(const Tensor& x, int K, int s0, int p0, int d0, ggml_type dst_type);
Tensor snake_1d(const Tensor& x, const Tensor& alpha);
Tensor col2im_1d(const Tensor& col, int s0, int oc, int p0);

}  // namespace mt
