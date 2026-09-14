using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 emotion encoder: a 4-layer Conformer operating on the raw 1024-dim
/// conditioning latents, producing 512-dim features for the perceiver.
///
/// Faithful to indextts/gpt/conformer_encoder.py with the 2.5 config
/// (output_size=512, linear_units=1024, heads=4, num_blocks=4, input_layer=conv2d2):
/// no macaron (ff_scale=1.0), relative-position attention (rel_shift is disabled
/// upstream), SiLU feed-forward, and a GLU + depthwise(k15) conv module.
/// </summary>
public static class Conformer
{
    public const int In = 1024;
    public const int D = 512;
    public const int Heads = 4;
    public const int HeadDim = 128;
    public const int FF = 1024;
    public const int K = 15;
    public const int Layers = 4;
    private const string P = "gpt.emo_conditioning_encoder.";

    internal static Tensor LnAffine(GgufWeights w, string p, Tensor x)
        => Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w[p + ".weight"]), w[p + ".bias"]);

    internal static Tensor Linear(GgufWeights w, string p, Tensor x)
        => Ops.Add(Ops.Linear(x, w[p + ".weight"]), w[p + ".bias"]);

    internal static Tensor ToHeads(Tensor x, int t, int n) => Attn.ToHeads(x, t, n, HeadDim);

    internal static Tensor FromHeads(Tensor x, int t, int n) => Attn.FromHeads(x, t, n, HeadDim);

    /// <summary>
    /// Depthwise conv1d (groups == channels). Built from im2col + a per-channel
    /// broadcast multiply + a K-sum, since vt's conv1d is dense and ggml's own
    /// ggml_conv_1d_dw is flagged unreliable. x PT [T, C]; w PT [C, 1, K].
    /// </summary>
    internal static Tensor DepthwiseConv1d(Tensor x, Tensor w, Tensor b, int stride, int pad, int dil)
    {
        int c = (int)w.Shape[0], k = (int)w.Shape[2];
        var cols = Ops.Im2colRafa(x, k, stride, pad, dil);            // PT [T_out, C*K]
        int tOut = (int)cols.Shape[0];
        var cols3 = Ops.Reshape(cols, new long[] { tOut, c, k });     // PT [T_out, C, K]
        var w2 = Ops.Reshape(w, new long[] { c, k });                 // PT [C, K] (broadcast)
        var summed = Ops.SumRows(Ops.Mul(cols3, w2));                 // PT [T_out, C]
        return Ops.Add(Ops.Reshape(summed, new long[] { tOut, c }), b);
    }

    /// <summary>Conformer convolution module: GLU → depthwise(k) → LN → SiLU → pointwise.</summary>
    internal static Tensor ConvModule(GgufWeights w, int li, Tensor x, int t)
    {
        string p = $"{P}encoders.{li}.conv_module.";
        var pw1 = Ops.Add(Ops.Conv1d(x, w[p + "pointwise_conv1.weight"]), w[p + "pointwise_conv1.bias"]);
        // GLU over the channel axis: [T, 2C] -> a * sigmoid(gate)
        ulong rb = (ulong)(2 * D * 4);
        var a = Ops.Contiguous(Ops.View2d(pw1, D, t, rb, 0));
        var gate = Ops.Contiguous(Ops.View2d(pw1, D, t, rb, (ulong)(D * 4)));
        var glu = Ops.Mul(a, Ops.Sigmoid(gate));

        var dw = DepthwiseConv1d(glu, w[p + "depthwise_conv.weight"], w[p + "depthwise_conv.bias"],
                                 1, (K - 1) / 2, 1);
        var normed = Ops.Silu(LnAffine(w, p + "norm", dw));
        return Ops.Add(Ops.Conv1d(normed, w[p + "pointwise_conv2.weight"]), w[p + "pointwise_conv2.bias"]);
    }

    /// <summary>Relative-position multi-head self-attention (no rel_shift). x,pos PT [T, D].</summary>
    internal static Tensor RelPosAttn(GgufWeights w, int li, Tensor x, Tensor pos, int t)
    {
        string p = $"{P}encoders.{li}.self_attn.";
        var q = ToHeads(Linear(w, p + "linear_q", x), t, Heads);
        var k = ToHeads(Linear(w, p + "linear_k", x), t, Heads);
        var v = ToHeads(Linear(w, p + "linear_v", x), t, Heads);

        var bu = Ops.Reshape(w[p + "pos_bias_u"], new long[] { Heads, 1, HeadDim });
        var bv = Ops.Reshape(w[p + "pos_bias_v"], new long[] { Heads, 1, HeadDim });
        var qu = Ops.Add(q, bu);
        var qv = Ops.Add(q, bv);

        var ph = ToHeads(Ops.Linear(pos, w[p + "linear_pos.weight"]), t, Heads);

        var ac = Ops.Bmm(qu, Ops.Transpose(k));                    // q @ k^T   -> PT [h, T, T]
        var bd = Ops.Bmm(qv, Ops.Transpose(ph));                   // q_v @ p^T -> PT [h, T, T]
        var scores = Ops.Scale(Ops.Add(ac, bd), 1f / MathF.Sqrt(HeadDim));
        var attn = Ops.SoftMax(scores);
        var o = Ops.Bmm(attn, v);                                  // PT [h, T, hd]
        return Linear(w, p + "linear_out", FromHeads(o, t, Heads));
    }

    /// <summary>Conv2dSubsampling2 + ReLU + Linear + pos scale/encoding. -> (y, pos) PT [T', D].</summary>
    internal static (Tensor y, Tensor pos) Embed(Graph g, GgufWeights w, Tensor x, int te)
    {
        var b4 = Ops.Reshape(x, new long[] { 1, 1, te, In });
        var cv = Ops.Conv2d(w[P + "embed.conv.0.weight"], b4, 2, 2, 0, 0, 1, 1);
        cv = Ops.Relu(Ops.Add(cv, Ops.Reshape(w[P + "embed.conv.0.bias"], new long[] { 1, D, 1, 1 })));

        // embed conv2d: k=3, s=2, p=0, d=1 -> out = (in - 3)/2 + 1
        int ow = (In - 3) / 2 + 1;   // 511
        int oh = (te - 3) / 2 + 1;

        var flat = Ops.Reshape(Ops.Contiguous(Ops.PermutePt(cv, 2, 1, 3, 0)),
                               new long[] { oh, D * ow });
        // out.0 contracts over 512*511 dims; an f16 weight makes the coopmat path
        // downcast the activation to f16, and the cancellation over this K turns that
        // into ~5% error. Casting the weight to f32 uses the f32 kernel instead (~0.08%).
        var wo = Ops.Cast(w[P + "embed.out.0.weight"], Ops.F32);
        var y = Ops.Scale(Ops.Add(Ops.Linear(flat, wo), w[P + "embed.out.0.bias"]), MathF.Sqrt(D));

        // pos_enc.pe is a fixed buffer: pull the first `oh` rows to host and feed
        // them as an F32 input (avoids a dtype cast + view over a computed node).
        var raw = w[P + "embed.pos_enc.pe"].ReadBytes();
        var posf = new float[oh * D];
        for (int i = 0; i < posf.Length; i++) posf[i] = (float)BitConverter.ToHalf(raw, i * 2);
        return (y, g.Input(new long[] { oh, D }, posf));
    }

    /// <summary>One pre-norm Conformer block (no macaron).</summary>
    internal static Tensor Layer(GgufWeights w, int li, Tensor x, Tensor pos, int t)
    {
        string p = $"{P}encoders.{li}.";
        x = Ops.Add(x, RelPosAttn(w, li, LnAffine(w, p + "norm_mha", x), pos, t));
        x = Ops.Add(x, ConvModule(w, li, LnAffine(w, p + "norm_conv", x), t));
        var ff = Linear(w, p + "feed_forward.w_2", Ops.Silu(Linear(w, p + "feed_forward.w_1",
                                                                  LnAffine(w, p + "norm_ff", x))));
        return LnAffine(w, p + "norm_final", Ops.Add(x, ff));
    }

    /// <summary>Raw 1024-dim conditioning latents PT [Te, In] -> features PT [T', D].</summary>
    public static Tensor Forward(Graph g, GgufWeights w, Tensor x, int te)
    {
        var (y, pos) = Embed(g, w, x, te);
        int t = (int)y.Shape[0];
        for (int li = 0; li < Layers; li++) y = Layer(w, li, y, pos, t);
        return LnAffine(w, P + "after_norm", y);
    }
}
