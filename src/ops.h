#pragma once

#include "tensor.h"

#include <vector>

namespace vt {

// All ops append nodes to the current Graph (see Graph::Scope) and return a
// handle. Nothing is computed until a Tensor is materialized (to_host).

Tensor matmul(const Tensor& a, const Tensor& b);  // a @ b, PyTorch semantics
Tensor mul_mat(const Tensor& a, const Tensor& b);  // raw ggml_mul_mat(a, b)

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float s);
Tensor scale_bias(const Tensor& a, float s, float b);  // a*s + b (scalars)

// View / shape ops
Tensor transpose(const Tensor& a);                                   // swap last two dims
Tensor contiguous(const Tensor& a);                                  // materialize a dense copy
Tensor reshape(const Tensor& a, const std::vector<int64_t>& shape);  // PyTorch-order shape
Tensor repeat(const Tensor& a, const std::vector<int64_t>& shape);

// Activations
Tensor gelu(const Tensor& a);
Tensor gelu_erf(const Tensor& a);  // exact erf GELU (matches torch F.gelu / nn.GELU())
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor elu(const Tensor& a);
Tensor silu(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor soft_max(const Tensor& a);
Tensor softplus(const Tensor& a);

// Elementwise math (IndexTTS / BigVGAN)
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor sqr(const Tensor& a);

// Linear / norms
Tensor linear(const Tensor& x, const Tensor& w);  // x @ w^T, w is [out, in]
Tensor rms_norm(const Tensor& a, float eps);
Tensor layer_norm(const Tensor& a, float eps);
Tensor group_norm(const Tensor& a, int n_groups, float eps);

// Attention masking: a[i, j] = -inf where j > i + n_past
Tensor diag_mask_inf(const Tensor& a, int n_past);

// Reductions / indexing / mixing
Tensor concat(const Tensor& a, const Tensor& b, int64_t pt_dim);
Tensor argmax(const Tensor& a);                    // along last dim -> I32 [.., 1]
Tensor get_rows(const Tensor& a, const Tensor& ids);
Tensor sum_rows(const Tensor& a);                  // sum along last dim
Tensor cast(const Tensor& a, ggml_type type);
Tensor cpy(const Tensor& a, const Tensor& dst);

// Scatter rows: dst[idx[i]] = src[i]. `dst` must be a contiguous-rows view (e.g.
// a layer slice of the KV cache); `src` has ne1 == len(idx). Use instead of cpy
// when the destination row depends on runtime state (n_past), so a captured
// graph stays valid across positions.
Tensor set_rows(const Tensor& dst, const Tensor& src, const Tensor& idx);

// Raw view / permute (ggml ne-space; escape hatch for layout-heavy code).
Tensor view_1d(const Tensor& a, int64_t ne0, size_t offset);
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
// depthwise conv1d (stock ggml_conv_1d_dw, groups == Cin). x: PT [T, C];
//   w: PT [C, 1, K] -> PT [T_out, C] (same layout as conv1d).
Tensor conv1d_dw(const Tensor& x, const Tensor& w, int stride, int pad, int dilation);
// conv2d (stock ggml_conv_2d). a (kernel): PT [OC, IC, KH, KW]; b (data): PT
//   [N, IC, IH, IW] -> PT [N, OC, OH, OW].
Tensor conv2d(const Tensor& a, const Tensor& b, int s0, int s1, int p0, int p1, int d0, int d1);
// conv2d with the output width split into ``n_tiles`` chunks so each chunk's im2col
// intermediate stays small (ggml_conv_2d materialises KH*KW*IC * OH*OW_tile). Exact
// same result as conv2d. ``n_tiles <= 0`` picks a count to keep each im2col under
// ~1.2 GiB. Use for large-kernel convs at high resolution, where the whole im2col
// would exceed the backend's per-buffer limit (e.g. Vulkan's 2 GiB).
Tensor conv2d_tiled(const Tensor& a, const Tensor& b, int s0, int s1, int p0, int p1, int d0,
                    int d1, int n_tiles = 0);
// conv_transpose_1d via w_perm + matmul + col2im_1d. w_perm is the pre-permuted
//   weight [K*OC, IC] (see convert_model: t.reshape(IC, K*OC).T). x:[T, Cin] ->
//   PT [OC, T_out]. The hand-written ggml_conv_transpose_1d op is numerically
//   wrong here and is not exposed.
Tensor conv_transpose_1d(const Tensor& x, const Tensor& w_perm, int stride, int oc);
Tensor im2col_rafa(const Tensor& x, int K, int s0, int p0, int d0, ggml_type dst_type);
Tensor snake_1d(const Tensor& x, const Tensor& alpha);
Tensor col2im_1d(const Tensor& col, int s0, int oc, int p0);

// depthwise conv2d (stock ggml_conv_2d_dw, groups == C). a (kernel): PT [C, 1, KH, KW];
//   b (data): PT [N, C, IH, IW] -> PT [N, C, OH, OW].
Tensor conv2d_dw(const Tensor& a, const Tensor& b, int s0, int s1, int p0, int p1, int d0, int d1);
// conv_transpose_2d (stock ggml_conv_transpose_2d_p0; no padding, single stride).
//   a (kernel): PT [IC, OC, KH, KW]; b (data): PT [N, IC, IH, IW] -> PT [N, OC, OH, OW].
Tensor conv_transpose_2d(const Tensor& a, const Tensor& b, int stride);
// 2D pooling. op: 0 = max, 1 = avg. a: PT [N, C, IH, IW] -> PT [N, C, OH, OW].
Tensor pool_2d(const Tensor& a, int op, int k0, int k1, int s0, int s1, float p0, float p1);
// nearest/lanczos upscale by an integer factor (stock ggml_upscale, mode: 0 nearest, 1 bilinear).
Tensor upsample(const Tensor& a, int scale_factor, int mode);
// zero-pad the END of each ggml dim (ne0=W, ne1=H, ...); p0 pads W, p1 pads H.
Tensor pad(const Tensor& a, int p0, int p1, int p2, int p3);
// elementwise clamp  min(a, max) -> [min_v, max_v].
Tensor clamp(const Tensor& a, float min_v, float max_v);

}  // namespace vt
