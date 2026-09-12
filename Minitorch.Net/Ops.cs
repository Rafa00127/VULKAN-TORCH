using System;

namespace Minitorch;

/// <summary>Op wrappers — every call appends to the current Graph (see Graph.Enter/Exit).</summary>
public static class Ops
{
    public const int F32 = 0;
    public const int F16 = 1;

    public static Tensor Matmul(Tensor a, Tensor b) => new(Native.mt_matmul(a.Handle, b.Handle));
    public static Tensor MulMat(Tensor a, Tensor b) => new(Native.mt_mul_mat(a.Handle, b.Handle));
    public static Tensor Add(Tensor a, Tensor b) => new(Native.mt_add(a.Handle, b.Handle));
    public static Tensor Sub(Tensor a, Tensor b) => new(Native.mt_sub(a.Handle, b.Handle));
    public static Tensor Mul(Tensor a, Tensor b) => new(Native.mt_mul(a.Handle, b.Handle));
    public static Tensor Scale(Tensor a, float s) => new(Native.mt_scale(a.Handle, s));
    public static Tensor Transpose(Tensor a) => new(Native.mt_transpose(a.Handle));
    public static Tensor Contiguous(Tensor a) => new(Native.mt_contiguous(a.Handle));
    public static Tensor Reshape(Tensor a, long[] shape) => new(Native.mt_reshape(a.Handle, shape, shape.Length));
    public static Tensor Repeat(Tensor a, long[] shape) => new(Native.mt_repeat(a.Handle, shape, shape.Length));
    public static Tensor Gelu(Tensor a) => new(Native.mt_gelu(a.Handle));
    public static Tensor Elu(Tensor a) => new(Native.mt_elu(a.Handle));
    public static Tensor Silu(Tensor a) => new(Native.mt_silu(a.Handle));
    public static Tensor Tanh(Tensor a) => new(Native.mt_tanh(a.Handle));
    public static Tensor SoftMax(Tensor a) => new(Native.mt_soft_max(a.Handle));
    public static Tensor Linear(Tensor x, Tensor w) => new(Native.mt_linear(x.Handle, w.Handle));
    public static Tensor RmsNorm(Tensor a, float eps) => new(Native.mt_rms_norm(a.Handle, eps));
    public static Tensor LayerNorm(Tensor a, float eps) => new(Native.mt_layer_norm(a.Handle, eps));
    public static Tensor Concat(Tensor a, Tensor b, long dim) => new(Native.mt_concat(a.Handle, b.Handle, dim));
    public static Tensor Argmax(Tensor a) => new(Native.mt_argmax(a.Handle));
    public static Tensor GetRows(Tensor a, Tensor ids) => new(Native.mt_get_rows(a.Handle, ids.Handle));
    public static Tensor SumRows(Tensor a) => new(Native.mt_sum_rows(a.Handle));
    public static Tensor Cast(Tensor a, int type) => new(Native.mt_cast(a.Handle, type));
    public static Tensor Cpy(Tensor a, Tensor dst) => new(Native.mt_cpy(a.Handle, dst.Handle));
    public static Tensor View2d(Tensor a, long ne0, long ne1, ulong nb1, ulong offset)
        => new(Native.mt_view_2d(a.Handle, ne0, ne1, (UIntPtr)nb1, (UIntPtr)offset));
    public static Tensor View3d(Tensor a, long ne0, long ne1, long ne2, ulong nb1, ulong nb2, ulong offset)
        => new(Native.mt_view_3d(a.Handle, ne0, ne1, ne2, (UIntPtr)nb1, (UIntPtr)nb2, (UIntPtr)offset));
    public static Tensor View4d(Tensor a, long ne0, long ne1, long ne2, long ne3, ulong nb1, ulong nb2, ulong nb3, ulong offset)
        => new(Native.mt_view_4d(a.Handle, ne0, ne1, ne2, ne3, (UIntPtr)nb1, (UIntPtr)nb2, (UIntPtr)nb3, (UIntPtr)offset));
    public static Tensor PermutePt(Tensor a, int p0, int p1, int p2, int p3)
        => new(Native.mt_permute_pt(a.Handle, p0, p1, p2, p3));

    public static Tensor Rope(Tensor a, Tensor? pos, int nDims, int mode = 2, int nCtxOrig = 0,
        float freqBase = 1e6f, float freqScale = 1f, float extFactor = 0f, float attnFactor = 1f,
        float betaFast = 32f, float betaSlow = 1f)
        => new(Native.mt_rope(a.Handle, pos?.Handle ?? IntPtr.Zero, nDims, mode, nCtxOrig,
            freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow));

    public static Tensor FlashAttn(Tensor q, Tensor k, Tensor v, Tensor? mask, float scale)
        => new(Native.mt_flash_attn(q.Handle, k.Handle, v.Handle, mask?.Handle ?? IntPtr.Zero,
            scale, 0f, 0f));

    public static Tensor Conv1d(Tensor x, Tensor w, int stride = 1, int pad = 0, int dilation = 1)
        => new(Native.mt_conv1d(x.Handle, w.Handle, stride, pad, dilation));
    public static Tensor ConvTranspose1d(Tensor x, Tensor wPerm, int stride, int oc)
        => new(Native.mt_conv_transpose_1d(x.Handle, wPerm.Handle, stride, oc));
    public static Tensor Snake1d(Tensor x, Tensor alpha) => new(Native.mt_snake_1d(x.Handle, alpha.Handle));
    public static Tensor Im2colRafa(Tensor x, int K, int s0, int p0, int d0)
        => new(Native.mt_im2col_rafa(x.Handle, K, s0, p0, d0));
    public static Tensor Col2im1d(Tensor col, int s0, int oc, int p0)
        => new(Native.mt_col2im_1d(col.Handle, s0, oc, p0));
}
