#include "ops.h"

#include "graph.h"

#include <stdexcept>

namespace mt {

static Graph& cur() {
    Graph* g = Graph::current();
    if (g == nullptr) {
        throw std::runtime_error("minitorch op called outside a Graph::Scope");
    }
    return *g;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2) throw std::runtime_error("matmul: expected 2D tensors");
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

Tensor scale(const Tensor& a, float s) {
    Graph& g = cur();
    ggml_tensor* r = ggml_scale(g.ctx(), a.raw(), s);
    g.add(r);
    return Tensor(r, &g);
}

Tensor transpose(const Tensor& a) {
    if (a.dim() != 2) throw std::runtime_error("transpose: only 2D supported");
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
    const int dim_ne = a.dim() - 1 - static_cast<int>(pt_dim);
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
    const int ax[4] = {3 - p3, 3 - p2, 3 - p1, 3 - p0};
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
    const int64_t K = w.shape()[2];
    const int64_t Cin = w.shape()[1];
    const int64_t Cout = w.shape()[0];
    ggml_tensor* cols = ggml_im2col_rafa(ctx, x.raw(), static_cast<int>(K), stride, pad, dilation,
                                         GGML_TYPE_F32);
    ggml_tensor* w2d = ggml_reshape_2d(ctx, w.raw(), Cin * K, Cout);
    ggml_tensor* y = ggml_mul_mat(ctx, w2d, cols);
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

}  // namespace mt
