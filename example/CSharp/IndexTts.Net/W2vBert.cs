using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 semantic encoder: Wav2Vec2-BERT (facebook/w2v-bert-2.0) on the SeamlessM4T
/// log-mel features. Its 18th hidden state (index 17), standardised by the checkpoint's
/// per-dimension mean/std, is the speaker conditioning fed to the s2mel length regulator.
/// Ported from transformers' Wav2Vec2BertModel.
///
/// 24 Conformer blocks: macaron FFN (half-step residual) -> relative_key self-attention ->
/// causal depthwise-conv module (GLU + kernel-31 depthwise conv) -> FFN -> final LayerNorm.
///
/// The relative-position term uses distance-embedding rows indexed by
/// clamp(pos_k - pos_q, -64, +8). That table depends only on the frame indices, so the whole
/// [T_q, T_k, 64] tensor is built once on the host and shared by every layer and head.
/// </summary>
public sealed class W2vBert
{
    private const int In = 160, D = 1024, Heads = 16, HD = 64, Inter = 4096, ConvK = 31;
    private const int LeftMax = 64, RightMax = 8;
    private const float Eps = 1e-5f;
    private const string P = "w2v.";
    private const int TakeLayer = 17;      // hidden_states[17] = output of layer 17 (1-based)

    private readonly GgufWeights _w;
    private Dictionary<string, Tensor>? _dbg;
    private Tensor _zero = null!, _attnBias = null!;

    public W2vBert(GgufWeights w) => _w = w;

    /// <summary>feats: PT [T, 160] -> PT [T, 1024] (the standardised hidden_states[17]).</summary>
    public Tensor Forward(Graph g, Tensor feats, float[] mask, float[] mean, float[] std,
                          Dictionary<string, Tensor>? dbg = null)
    {
        _dbg = dbg;
        int t = (int)feats.Shape[0];
        // The feature extractor pads to a multiple of the stride, so the trailing frame(s) are
        // marked 0. The reference zeroes them and masks them out of attention; without that the
        // padding leaks into every frame through the causal depthwise convolution.
        _zero = g.Input(new long[] { t, 1 }, mask);
        var bias = new float[t];
        for (int i = 0; i < t; i++) bias[i] = mask[i] != 0f ? 0f : -1e9f;
        _attnBias = g.Input(new long[] { 1, 1, t }, bias);

        var x = Ops.Mul(Projection(g, feats), _zero);
        if (_dbg != null) _dbg["hs0"] = x;
        var distIdx = DistIndex(g, t);
        for (int i = 0; i < TakeLayer; i++)
        {
            x = Layer(g, x, i, RelativeTable(g, i, t, distIdx));
            if (_dbg != null && (i == 0 || i == 1 || i == 4 || i == 8 || i == 16))
                _dbg[$"hs{i + 1}"] = x;
        }
        // (x - mean) / std, per hidden dimension.
        var mu = g.Input(new long[] { 1, D }, mean);
        var sd = g.Input(new long[] { 1, D }, std);
        return Ops.Div(Ops.Sub(x, mu), sd);
    }

    /// <summary>Feature projection: LayerNorm(160) then Linear(160 -> 1024).</summary>
    private Tensor Projection(Graph g, Tensor x)
    {
        string p = P + "feature_projection.";
        x = AffineLn(g, x, p + "layer_norm");
        return Ops.Add(Ops.Linear(x, _w[p + "projection.weight"]), _w[p + "projection.bias"]);
    }

    /// <summary>LayerNorm with weight/bias (ggml_norm itself is affine-free).</summary>
    private Tensor AffineLn(Graph g, Tensor x, string prefix)
        => Ops.Add(Ops.Mul(Ops.LayerNorm(x, Eps), _w[prefix + ".weight"]), _w[prefix + ".bias"]);

    private Tensor Layer(Graph g, Tensor x, int i, Tensor pe)
    {
        string p = P + $"encoder.layers.{i}.";
        // 1. FFN1, half-step residual.
        var r = x;
        x = Ops.Add(Ops.Scale(FFN(g, AffineLn(g, x, p + "ffn1_layer_norm"),
                                  p + "ffn1.intermediate_dense", p + "ffn1.output_dense"), 0.5f), r);
        // 2. Self-attention.
        r = x;
        x = Ops.Add(Attention(g, AffineLn(g, x, p + "self_attn_layer_norm"), i, pe), r);
        // 3. Convolution module.
        r = x;
        x = Ops.Add(ConvModule(g, x, p + "conv_module."), r);
        // 4. FFN2, half-step residual, then the final LayerNorm.
        r = x;
        x = Ops.Add(Ops.Scale(FFN(g, AffineLn(g, x, p + "ffn2_layer_norm"),
                                  p + "ffn2.intermediate_dense", p + "ffn2.output_dense"), 0.5f), r);
        return AffineLn(g, x, p + "final_layer_norm");
    }

    private Tensor FFN(Graph g, Tensor x, string inter, string outp)
    {
        var h = Ops.Silu(Ops.Add(Ops.Linear(x, _w[inter + ".weight"]), _w[inter + ".bias"]));
        return Ops.Add(Ops.Linear(h, _w[outp + ".weight"]), _w[outp + ".bias"]);
    }

    /// <summary>relative_key self-attention. x: PT [T, 1024] -> PT [T, 1024].</summary>
    private Tensor Attention(Graph g, Tensor x, int i, Tensor pe)
    {
        string p = P + $"encoder.layers.{i}.self_attn.";
        int t = (int)x.Shape[0];
        var q = Ops.Add(Ops.Linear(x, _w[p + "linear_q.weight"]), _w[p + "linear_q.bias"]);
        var k = Ops.Add(Ops.Linear(x, _w[p + "linear_k.weight"]), _w[p + "linear_k.bias"]);
        var v = Ops.Add(Ops.Linear(x, _w[p + "linear_v.weight"]), _w[p + "linear_v.bias"]);

        float inv = 1f / MathF.Sqrt(HD);
        var qh = Attn.ToHeads(q, t, Heads, HD);                       // [H, T, HD]
        var kh = Attn.ToHeads(k, t, Heads, HD);
        var vh = Attn.ToHeads(v, t, Heads, HD);
        // q @ k^T, with k transposed first (Bmm contracts the last dim of each operand).
        var scores = Ops.Scale(Ops.Bmm(qh, Ops.Transpose(kh)), inv);  // [H, T, T]

        // Relative term: rel[q, h, k] = sum_d q[q,h,d] * pe[q,k,d]. mul_mat batches over the
        // query index, so it comes out as PT [T, T, H]; permute it to [H, T, T].
        var q3 = Ops.Reshape(q, new long[] { t, Heads, HD });         // ne=[HD,H,T]
        var rel = Ops.MulMat(q3, pe);                                 // ne=[H,T,T] -> PT [T,T,H]
        var rel4 = Ops.PermutePt(Ops.Reshape(rel, new long[] { t, t, Heads, 1 }), 2, 0, 1, 3);
        rel = Ops.Reshape(Ops.Contiguous(rel4), new long[] { Heads, t, t });
        scores = Ops.Add(scores, Ops.Scale(rel, inv));
        scores = Ops.Add(scores, _attnBias);                          // mask padded keys

        var probs = Ops.SoftMax(scores);
        var o = Ops.Bmm(probs, vh);                                   // [H, T, HD]
        return Ops.Add(Ops.Linear(Attn.FromHeads(o, t, Heads, HD), _w[p + "linear_out.weight"]),
                       _w[p + "linear_out.bias"]);
    }

    /// <summary>Conformer convolution: LayerNorm -> GLU(1x1) -> causal depthwise(k31) ->
    /// LayerNorm -> swish -> 1x1. x: PT [T, 1024].</summary>
    private Tensor ConvModule(Graph g, Tensor x, string p)
    {
        int t = (int)x.Shape[0];
        x = Ops.Mul(AffineLn(g, x, p + "layer_norm"), _zero);
        var pw = Ops.Conv1d(x, _w[p + "pointwise_conv1.weight"], 1, 0, 1);   // [T, 2048]
        var a = Slice(g, pw, t, 2 * D, 0, D);
        var b = Slice(g, pw, t, 2 * D, D, D);
        var glu = Ops.Mul(a, Ops.Sigmoid(b));                                // [T, 1024]
        // Causal: pad (K-1) zeros on the left, then a zero-padded-free depthwise conv.
        var padded = Ops.Concat(Zero(g, ConvK - 1, D), glu, 0);              // [T+30, 1024]
        var dw = Ops.Conv1dDw(padded, _w[p + "depthwise_conv.weight"], 1, 0, 1);  // [T, 1024]
        dw = AffineLn(g, dw, p + "depthwise_layer_norm");
        dw = Ops.Silu(dw);
        return Ops.Conv1d(dw, _w[p + "pointwise_conv2.weight"], 1, 0, 1);
    }

    /// <summary>clamp(pos_k - pos_q, -64, +8) + 64 for every (q, k) pair — shared by all layers.</summary>
    private Tensor DistIndex(Graph g, int t)
    {
        var idx = new int[t * t];
        for (int qq = 0; qq < t; qq++)
            for (int kk = 0; kk < t; kk++)
                idx[qq * t + kk] = Math.Clamp(kk - qq, -LeftMax, RightMax) + LeftMax;
        return g.InputI32(new long[] { t * t }, idx);
    }

    /// <summary>The [T_q, T_k, 64] relative-position table for one layer. Each layer owns its own
    /// distance embedding, so the table cannot be shared across layers.</summary>
    private Tensor RelativeTable(Graph g, int i, int t, Tensor distIdx)
    {
        var emb = _w[P + $"encoder.layers.{i}.self_attn.distance_embedding.weight"];  // PT [73, 64]
        var rows = Ops.GetRows(emb, distIdx);                       // [T*T, 64]
        return Ops.Reshape(rows, new long[] { t, t, HD });
    }

    /// <summary>Channel slice of a PT [T, total] tensor, materialized.</summary>
    private static Tensor Slice(Graph g, Tensor x, int t, int total, int from, int count)
        => Ops.Contiguous(Ops.View2d(x, count, t, (ulong)(total * 4), (ulong)(from * 4)));

    private static Tensor Zero(Graph g, int t, int c) => g.Input(new long[] { t, c }, new float[t * c]);

}
