using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS GPT backbone: HF `GPT2Model` (24 pre-LN layers, Conv1D weights) followed
/// by `ln_f`, the extra `final_norm` and `mel_head`.
///
/// Note the weight layouts: GPT2's c_attn / c_proj / c_fc / c_proj are HF Conv1D
/// ([in, out], so `x @ W`), while mel_head / text_head are nn.Linear ([out, in],
/// so `x @ W^T` via Ops.Linear).
/// </summary>
public static class Gpt2
{
    public const int D = 1280;
    public const int Heads = 20;
    public const int HeadDim = 64;
    public const int Layers = 24;
    public const int MelVocab = 8194;

    // PT [T, n*hd] -> PT [n, T, hd]
    public static Tensor ToHeads(Tensor x, int t, int n) => VulkanTorch.Attn.ToHeads(x, t, n, HeadDim);

    /// <summary>Causal additive mask, PT [t, t]: 0 where key &lt;= query, else -inf.</summary>
    public static float[] CausalMask(int t)
    {
        var m = new float[(long)t * t];
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                m[q * t + k] = k <= q ? 0f : float.NegativeInfinity;
        return m;
    }

    /// <summary>Raw attention output before c_proj, PT [t, D] (diagnostics).</summary>
    public static Tensor AttnRaw(GgufWeights w, int li, Tensor h, Tensor mask, int t)
    {
        string p = $"gpt.gpt.h.{li}.";
        var qkv = Ops.Add(Ops.Matmul(h, w[p + "attn.c_attn.weight"]), w[p + "attn.c_attn.bias"]);
        ulong rowBytes = (ulong)(3 * D * 4);
        var q = ToHeads(Ops.Contiguous(Ops.View2d(qkv, D, t, rowBytes, 0)), t, Heads);
        var k = ToHeads(Ops.Contiguous(Ops.View2d(qkv, D, t, rowBytes, (ulong)(D * 4))), t, Heads);
        var v = ToHeads(Ops.Contiguous(Ops.View2d(qkv, D, t, rowBytes, (ulong)(2 * D * 4))), t, Heads);
        var attn = Ops.FlashAttn(q, k, v, mask, 1f / MathF.Sqrt(HeadDim));
        return Ops.Reshape(Ops.Contiguous(attn), new long[] { t, D });
    }

    /// <summary>Self-attention block output (pre-residual), PT [t, D].</summary>
    public static Tensor Attn(GgufWeights w, int li, Tensor h, Tensor mask, int t)
    {
        string p = $"gpt.gpt.h.{li}.";
        var attn = AttnRaw(w, li, h, mask, t);
        return Ops.Add(Ops.Matmul(attn, w[p + "attn.c_proj.weight"]), w[p + "attn.c_proj.bias"]);
    }

    /// <summary>MLP block output (pre-residual), PT [t, D].</summary>
    public static Tensor Mlp(GgufWeights w, int li, Tensor h)
    {
        string p = $"gpt.gpt.h.{li}.";
        var fc = Ops.Gelu(Ops.Add(Ops.Matmul(h, w[p + "mlp.c_fc.weight"]), w[p + "mlp.c_fc.bias"]));
        return Ops.Add(Ops.Matmul(fc, w[p + "mlp.c_proj.weight"]), w[p + "mlp.c_proj.bias"]);
    }

    public static Tensor Layer(GgufWeights w, int li, Tensor x, Tensor mask, int t)
    {
        string p = $"gpt.gpt.h.{li}.";
        var h1 = Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w[p + "ln_1.weight"]), w[p + "ln_1.bias"]);
        var x2 = Ops.Add(x, Attn(w, li, h1, mask, t));

        var h2 = Ops.Add(Ops.Mul(Ops.LayerNorm(x2, 1e-5f), w[p + "ln_2.weight"]), w[p + "ln_2.bias"]);
        return Ops.Add(x2, Mlp(w, li, h2));
    }

    /// <summary>inputs_embeds PT [t, D] -> (after ln_f, after final_norm).</summary>
    public static (Tensor, Tensor) Forward(GgufWeights w, Graph g, Tensor emb, int t)
    {
        var mask = Ops.Cast(g.Input(new long[] { t, t }, CausalMask(t)), Ops.F16);
        var x = emb;
        for (int li = 0; li < Layers; li++) x = Layer(w, li, x, mask, t);

        var lastHidden = Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w["gpt.gpt.ln_f.weight"]),
                                 w["gpt.gpt.ln_f.bias"]);
        var finalNorm = Ops.Add(Ops.Mul(Ops.LayerNorm(lastHidden, 1e-5f), w["gpt.final_norm.weight"]),
                                w["gpt.final_norm.bias"]);
        return (lastHidden, finalNorm);
    }

    /// <summary>final_norm output -> mel logits PT [t, MelVocab] (nn.Linear layout).</summary>
    public static Tensor MelHead(GgufWeights w, Tensor hidden)
        => Ops.Add(Ops.Linear(hidden, w["gpt.mel_head.weight"]), w["gpt.mel_head.bias"]);
}
