#include "ops.h"

#include "graph.h"

#include <stdexcept>

namespace vt {

static Graph& cur() {
    Graph* g = Graph::current();
    if (g == nullptr) {
        throw std::runtime_error("vulkantorch op called outside a Graph::Scope");
    }
    return *g;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // ggml_n_dims collapses trailing PT dims of size 1, so a PT [1, K] operand reports
    // rank 1. Check the actual ggml shape instead: a matrix is ne[2] == ne[3] == 1.
    if (!ggml_is_matrix(a.raw()) || !ggml_is_matrix(b.raw()))
        throw std::runtime_error("matmul: expected 2D tensors");
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    // C = A @ B. With the ne-reversed layout: b^T is K-contiguous, and
    // ggml_mul_mat(b^T, a) yields a node laid out as PyTorch [M, N].
    ggml_tensor* bt = ggml_cont(ctx, ggml_transpose(ctx, b.raw()));
    ggml_tensor* c = ggml_mul_mat(ctx, bt, a.raw());
    g.add(c);
    return Tensor(c, &g);
}

Tensor mul_mat(const Tensor& a, const Tensor& b) {
    Graph& g = cur();
    ggml_tensor* r = ggml_mul_mat(g.ctx(), a.raw(), b.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor add(const Tensor& a, const Tensor& b) {
    Graph& g = cur();
    ggml_tensor* r = ggml_add(g.ctx(), a.raw(), b.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Graph& g = cur();
    ggml_tensor* r = ggml_sub(g.ctx(), a.raw(), b.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Graph& g = cur();
    ggml_tensor* r = ggml_mul(g.ctx(), a.raw(), b.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor div(const Tensor& a, const Tensor& b) {
    Graph& g = cur();
    ggml_tensor* r = ggml_div(g.ctx(), a.raw(), b.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor scale(const Tensor& a, float s) {
    Graph& g = cur();
    ggml_tensor* r = ggml_scale(g.ctx(), a.raw(), s);
    g.add(r);
    return Tensor(r, &g);
}

Tensor scale_bias(const Tensor& a, float s, float b) {
    Graph& g = cur();
    ggml_tensor* r = ggml_scale_bias(g.ctx(), a.raw(), s, b);
    g.add(r);
    return Tensor(r, &g);
}

Tensor transpose(const Tensor& a) {
    // ggml_transpose swaps ne0/ne1, i.e. PyTorch transpose(-2, -1) for any rank.
    Graph& g = cur();
    ggml_tensor* r = ggml_transpose(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor contiguous(const Tensor& a) {
    if (ggml_is_contiguous(a.raw())) return a;
    Graph& g = cur();
    ggml_tensor* r = ggml_cont(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& shape) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    std::vector<int64_t> ne(shape.rbegin(), shape.rend());
    ggml_tensor* r = nullptr;
    switch (static_cast<int>(ne.size())) {
        case 1: r = ggml_reshape_1d(ctx, a.raw(), ne[0]); break;
        case 2: r = ggml_reshape_2d(ctx, a.raw(), ne[0], ne[1]); break;
        case 3: r = ggml_reshape_3d(ctx, a.raw(), ne[0], ne[1], ne[2]); break;
        case 4: r = ggml_reshape_4d(ctx, a.raw(), ne[0], ne[1], ne[2], ne[3]); break;
        default: throw std::runtime_error("reshape: only 1..4 dims supported");
    }
    g.add(r);
    return Tensor(r, &g);
}

Tensor repeat(const Tensor& a, const std::vector<int64_t>& shape) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    ggml_tensor* shape_t = new_tensor_pt(ctx, a.raw()->type, shape);
    ggml_tensor* r = ggml_repeat(ctx, a.raw(), shape_t);
    g.add(r);
    return Tensor(r, &g);
}

Tensor gelu(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_gelu(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor gelu_erf(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_gelu_erf(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor relu(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_relu(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}
Tensor sigmoid(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_sigmoid(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}
Tensor exp(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_exp(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}
Tensor elu(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_elu(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor silu(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_silu(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor tanh(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_tanh(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor soft_max(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_soft_max(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor softplus(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_softplus(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor sin(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_sin(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor cos(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_cos(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor sqrt(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_sqrt(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor sqr(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_sqr(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor group_norm(const Tensor& a, int n_groups, float eps) {
    Graph& g = cur();
    ggml_tensor* r = ggml_group_norm(g.ctx(), a.raw(), n_groups, eps);
    g.add(r);
    return Tensor(r, &g);
}

Tensor diag_mask_inf(const Tensor& a, int n_past) {
    Graph& g = cur();
    ggml_tensor* r = ggml_diag_mask_inf(g.ctx(), a.raw(), n_past);
    g.add(r);
    return Tensor(r, &g);
}

Tensor linear(const Tensor& x, const Tensor& w) {
    Graph& g = cur();
    ggml_tensor* r = ggml_mul_mat(g.ctx(), w.raw(), x.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor rms_norm(const Tensor& a, float eps) {
    Graph& g = cur();
    ggml_tensor* r = ggml_rms_norm(g.ctx(), a.raw(), eps);
    g.add(r);
    return Tensor(r, &g);
}

Tensor layer_norm(const Tensor& a, float eps) {
    Graph& g = cur();
    ggml_tensor* r = ggml_norm(g.ctx(), a.raw(), eps);
    g.add(r);
    return Tensor(r, &g);
}

Tensor concat(const Tensor& a, const Tensor& b, int64_t pt_dim) {
    Graph& g = cur();
    // Use the larger rank: ggml_n_dims collapses trailing PT dims of size 1, so a
    // [1, D] operand reports rank 1 and would otherwise pick the wrong ggml axis.
    const int rank = a.dim() > b.dim() ? a.dim() : b.dim();
    const int dim_ne = rank - 1 - static_cast<int>(pt_dim);
    ggml_tensor* r = ggml_concat(g.ctx(), a.raw(), b.raw(), dim_ne);
    g.add(r);
    return Tensor(r, &g);
}

Tensor argmax(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_argmax(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor get_rows(const Tensor& a, const Tensor& ids) {
    Graph& g = cur();
    ggml_tensor* r = ggml_get_rows(g.ctx(), a.raw(), ids.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor sum_rows(const Tensor& a) {
    Graph& g = cur();
    ggml_tensor* r = ggml_sum_rows(g.ctx(), a.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor cast(const Tensor& a, ggml_type type) {
    Graph& g = cur();
    ggml_tensor* r = ggml_cast(g.ctx(), a.raw(), type);
    g.add(r);
    return Tensor(r, &g);
}

Tensor cpy(const Tensor& a, const Tensor& dst) {
    Graph& g = cur();
    ggml_tensor* r = ggml_cpy(g.ctx(), a.raw(), dst.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor set_rows(const Tensor& dst, const Tensor& src, const Tensor& idx) {
    Graph& g = cur();
    ggml_tensor* r = ggml_set_rows(g.ctx(), dst.raw(), src.raw(), idx.raw());
    g.add(r);
    return Tensor(r, &g);
}

Tensor view_1d(const Tensor& a, int64_t ne0, size_t offset) {
    Graph& g = cur();
    ggml_tensor* r = ggml_view_1d(g.ctx(), a.raw(), ne0, offset);
    g.add(r);
    return Tensor(r, &g);
}

Tensor view_2d(const Tensor& a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
    Graph& g = cur();
    ggml_tensor* r = ggml_view_2d(g.ctx(), a.raw(), ne0, ne1, nb1, offset);
    g.add(r);
    return Tensor(r, &g);
}

Tensor view_3d(const Tensor& a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2,
               size_t offset) {
    Graph& g = cur();
    ggml_tensor* r = ggml_view_3d(g.ctx(), a.raw(), ne0, ne1, ne2, nb1, nb2, offset);
    g.add(r);
    return Tensor(r, &g);
}

Tensor view_4d(const Tensor& a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1,
               size_t nb2, size_t nb3, size_t offset) {
    Graph& g = cur();
    ggml_tensor* r = ggml_view_4d(g.ctx(), a.raw(), ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    g.add(r);
    return Tensor(r, &g);
}

Tensor permute_pt(const Tensor& a, int p0, int p1, int p2, int p3) {
    Graph& g = cur();
    // PyTorch semantics: result dim j == input dim p[j]. ggml_permute(a, axis0..3)
    // instead scatters (ne[axis_i] = a.ne[i]), so invert p before mapping.
    const int p[4] = {p0, p1, p2, p3};
    int q[4];
    for (int k = 0; k < 4; ++k) q[p[k]] = k;
    const int ax[4] = {3 - q[3], 3 - q[2], 3 - q[1], 3 - q[0]};
    ggml_tensor* r = ggml_permute(g.ctx(), a.raw(), ax[0], ax[1], ax[2], ax[3]);
    g.add(r);
    return Tensor(r, &g);
}

Tensor rope(const Tensor& a, const Tensor& pos, int n_dims, int mode, int n_ctx_orig,
            float freq_base, float freq_scale, float ext_factor, float attn_factor,
            float beta_fast, float beta_slow) {
    Graph& g = cur();
    ggml_tensor* p = pos.defined() ? pos.raw() : nullptr;
    ggml_tensor* r = ggml_rope_ext(g.ctx(), a.raw(), p, nullptr, n_dims, mode, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast,
                                   beta_slow);
    g.add(r);
    return Tensor(r, &g);
}

Tensor flash_attn(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask,
                  float scale, float max_bias, float logit_softcap) {
    Graph& g = cur();
    ggml_tensor* m = mask.defined() ? mask.raw() : nullptr;
    ggml_tensor* r = ggml_flash_attn_ext(g.ctx(), q.raw(), k.raw(), v.raw(), m, scale, max_bias,
                                         logit_softcap);
    g.add(r);
    return Tensor(r, &g);
}

Tensor conv1d(const Tensor& x, const Tensor& w, int stride, int pad, int dilation) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    // x: [T, Cin] (ne=[Cin, T]); w: [Cout, Cin, K] (ne=[K, Cin, Cout]).
    // Read ne[] directly: Tensor::shape() collapses size-1 dims, so a 1-channel conv
    // (Cout == 1) reports rank 2 and loses K.
    const int64_t K = w.raw()->ne[0];
    const int64_t Cin = w.raw()->ne[1];
    const int64_t Cout = w.raw()->ne[2];
    ggml_tensor* cols = ggml_im2col_rafa(ctx, x.raw(), static_cast<int>(K), stride, pad, dilation,
                                         GGML_TYPE_F32);
    ggml_tensor* w2d = ggml_reshape_2d(ctx, w.raw(), Cin * K, Cout);
    ggml_tensor* y = ggml_mul_mat(ctx, w2d, cols);
    g.add(y);
    return Tensor(y, &g);
}

Tensor conv2d(const Tensor& a, const Tensor& b, int s0, int s1, int p0, int p1, int d0, int d1) {
    Graph& g = cur();
    ggml_tensor* r = ggml_conv_2d(g.ctx(), a.raw(), b.raw(), s0, s1, p0, p1, d0, d1);
    g.add(r);
    return Tensor(r, &g);
}

Tensor conv2d_tiled(const Tensor& a, const Tensor& b, int s0, int s1, int p0, int p1, int d0,
                    int d1, int n_tiles) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    ggml_tensor* k = a.raw();
    ggml_tensor* bt = b.raw();
    const int64_t IW = bt->ne[0], IH = bt->ne[1], IC = bt->ne[2], N = bt->ne[3];
    const int64_t KW = k->ne[0], KH = k->ne[1];
    const int64_t K_eff = (KW - 1) * d0 + 1;
    const int64_t OW = (IW + 2 * p0 - K_eff) / s0 + 1;

    // auto tile count: keep each chunk's im2col (KH*KW*IC * OH*OW_tile floats) small
    if (n_tiles <= 0) {
        const int64_t per_col = KH * KW * IC * IH * (int64_t)sizeof(float);
        const int64_t budget = 256LL << 20;  // ~256 MiB per chunk's im2col
        n_tiles = (int)((OW * per_col + budget - 1) / budget);
        if (n_tiles < 1) n_tiles = 1;
    }
    if (n_tiles <= 1 || OW <= 0) return conv2d(a, b, s0, s1, p0, p1, d0, d1);

    // xp: input zero-padded left+right in W -> [IW + 2*p0, IH, IC, N]; conv then uses p=0
    ggml_tensor* xp = bt;
    if (p0 > 0) {
        // zeros of shape [p0, IH, IC, N] = scale(view(bt), 0) (view must be made contiguous first)
        ggml_tensor* zw = ggml_cont(ctx, ggml_view_4d(ctx, bt, p0, IH, IC, N, bt->nb[1],
                                                      bt->nb[2], bt->nb[3], 0));
        ggml_tensor* z = ggml_scale(ctx, zw, 0.0f);
        xp = ggml_pad(ctx, ggml_concat(ctx, z, bt, 0), p0, 0, 0, 0);
        g.add(xp);
    }

    const int64_t per = (OW + n_tiles - 1) / n_tiles;
    ggml_tensor* out = nullptr;
    for (int64_t ow0 = 0; ow0 < OW; ow0 += per) {
        const int64_t ow1 = std::min(ow0 + per, OW);
        const int64_t in_w = (ow1 - ow0 - 1) * s0 + K_eff;   // slice width needed
        ggml_tensor* bs = ggml_cont(ctx, ggml_view_4d(ctx, xp, in_w, IH, IC, N, xp->nb[1],
                                                      xp->nb[2], xp->nb[3],
                                                      (size_t)(ow0 * s0) * xp->nb[0]));
        ggml_tensor* c = ggml_conv_2d(ctx, k, bs, s0, s1, 0, p1, d0, d1);
        if (out == nullptr) {
            out = c;
        } else {
            out = ggml_concat(ctx, out, c, 0);
        }
        g.add(out);
    }
    return Tensor(out, &g);
}

Tensor conv1d_dw(const Tensor& x, const Tensor& w, int stride, int pad, int dilation) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    // x: PT [T, C]; w: PT [C, 1, K] -> PT [T_out, C].
    // ggml_conv_1d_dw is unreliable (its own header warns about it); lower to the plain
    // depthwise-2d kernel with a height of 1, exactly as audio.cpp does.
    // ggml_conv_2d_dw_direct takes a: ne=[KW,KH,1,C] (PT [C,1,KH,KW]) and
    // b: ne=[W,H,C,N] (PT [N,C,H,W]) -> ne=[OW,OH,C,N] (PT [N,C,OH,OW]).
    const int64_t T = x.shape()[0];
    const int64_t C = w.shape()[0];
    const int64_t K = w.shape()[2];
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x.raw()));  // PT [C, T]
    ggml_tensor* b = ggml_reshape_4d(ctx, xt, T, 1, C, 1);           // PT [1, C, 1, T]
    ggml_tensor* a = ggml_reshape_4d(ctx, w.raw(), K, 1, 1, C);      // PT [C, 1, 1, K]
    ggml_tensor* y4 = ggml_conv_2d_dw_direct(ctx, a, b, stride, 1, pad, 0, dilation, 1);
    ggml_tensor* y2 = ggml_reshape_2d(ctx, y4, y4->ne[0], C);        // PT [C, T_out]
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, y2));        // PT [T_out, C]
    g.add(y);
    return Tensor(y, &g);
}
Tensor conv_transpose_1d(const Tensor& x, const Tensor& w_perm, int stride, int oc) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    // x: PT [T, Cin] (ne [Cin, T]); w_perm: PT [K*OC, IC] (ne [IC, K*OC]).
    // mul_mat contracts IC -> col [K*OC, T]; col2im gathers -> [T_raw, OC] (ne)
    // i.e. PT [OC, T_raw].
    ggml_tensor* col = ggml_mul_mat(ctx, w_perm.raw(), x.raw());
    ggml_tensor* y = ggml_col2im_1d(ctx, col, stride, oc, 0);
    g.add(y);
    return Tensor(y, &g);
}

Tensor im2col_rafa(const Tensor& x, int K, int s0, int p0, int d0, ggml_type dst_type) {
    Graph& g = cur();
    ggml_tensor* r = ggml_im2col_rafa(g.ctx(), x.raw(), K, s0, p0, d0, dst_type);
    g.add(r);
    return Tensor(r, &g);
}

Tensor snake_1d(const Tensor& x, const Tensor& alpha) {
    Graph& g = cur();
    ggml_context* ctx = g.ctx();
    // ggml_snake_1d expects x with ne0 = T (per-channel alpha aligns on ne1).
    // Our API is PT [T, C] (ne0 = C), so transpose in and back out.
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x.raw()));  // ne0 = T
    ggml_tensor* y = ggml_snake_1d(ctx, xt, alpha.raw());
    ggml_tensor* yt = ggml_cont(ctx, ggml_transpose(ctx, y));        // back to PT [T, C]
    g.add(yt);
    return Tensor(yt, &g);
}

Tensor col2im_1d(const Tensor& col, int s0, int oc, int p0) {
    Graph& g = cur();
    ggml_tensor* r = ggml_col2im_1d(g.ctx(), col.raw(), s0, oc, p0);
    g.add(r);
    return Tensor(r, &g);
}

Tensor conv2d_dw(const Tensor& a, const Tensor& b, int s0, int s1, int p0, int p1, int d0, int d1) {
    Graph& g = cur();
    // Direct kernel: a ne=[KW,KH,1,C] (PT [C,1,KH,KW]), b ne=[W,H,C,N] (PT [N,C,H,W]).
    ggml_tensor* r = ggml_conv_2d_dw_direct(g.ctx(), a.raw(), b.raw(), s0, s1, p0, p1, d0, d1);
    g.add(r);
    return Tensor(r, &g);
}

Tensor conv_transpose_2d(const Tensor& a, const Tensor& b, int stride) {
    Graph& g = cur();
    ggml_tensor* r = ggml_conv_transpose_2d_p0(g.ctx(), a.raw(), b.raw(), stride);
    g.add(r);
    return Tensor(r, &g);
}

Tensor pool_2d(const Tensor& a, int op, int k0, int k1, int s0, int s1, float p0, float p1) {
    Graph& g = cur();
    ggml_tensor* r = ggml_pool_2d(g.ctx(), a.raw(), static_cast<ggml_op_pool>(op), k0, k1, s0, s1,
                                  p0, p1);
    g.add(r);
    return Tensor(r, &g);
}

Tensor upsample(const Tensor& a, int scale_factor, int mode) {
    Graph& g = cur();
    ggml_tensor* r = ggml_upscale(g.ctx(), a.raw(), scale_factor,
                                  static_cast<ggml_scale_mode>(mode));
    g.add(r);
    return Tensor(r, &g);
}

Tensor pad(const Tensor& a, int p0, int p1, int p2, int p3) {
    Graph& g = cur();
    ggml_tensor* r = ggml_pad(g.ctx(), a.raw(), p0, p1, p2, p3);
    g.add(r);
    return Tensor(r, &g);
}

Tensor clamp(const Tensor& a, float min_v, float max_v) {
    Graph& g = cur();
    ggml_tensor* r = ggml_clamp(g.ctx(), a.raw(), min_v, max_v);
    g.add(r);
    return Tensor(r, &g);
}

}  // namespace vt
