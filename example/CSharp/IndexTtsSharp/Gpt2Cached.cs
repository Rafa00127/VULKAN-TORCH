using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// KV-cached GPT2 decode for IndexTTS 2.5. Uses the shared <see cref="KvCache"/> +
/// <see cref="GraphCache"/> helpers: one captured graph per window bucket, K/V written at
/// the runtime position via set_rows, and attention reading a fixed window whose
/// unwritten slots are -inf in the mask. Each decode step becomes one token instead of
/// a full 24-layer recompute.
/// </summary>
public static class Gpt2Cached
{
    public const int D = Gpt2.D, NH = Gpt2.Heads, HD = Gpt2.HeadDim, NL = Gpt2.Layers;

    public static KvCache NewCache(Device dev, int maxCtx)
        => new KvCache(dev, NL, NH, HD, maxCtx);

    /// <summary>One GPT2 block reading/writing the cache. `pos` = runtime positions (I32),
    /// `mask` = F16 PT [t, lk].</summary>
    public static Tensor LayerCached(GgufWeights w, int li, Tensor x, KvCache kv, Tensor pos,
                                     Tensor mask, int lk, int t)
    {
        string p = $"gpt.gpt.h.{li}.";
        var residual = x;
        var h = Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w[p + "ln_1.weight"]), w[p + "ln_1.bias"]);

        var qkv = Ops.Add(Ops.Matmul(h, w[p + "attn.c_attn.weight"]), w[p + "attn.c_attn.bias"]);
        ulong rb = (ulong)(3 * D * 4);
        var q = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, D, t, rb, 0)), t, NH);
        var k = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, D, t, rb, (ulong)(D * 4))), t, NH);
        var v = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, D, t, rb, (ulong)(2 * D * 4))), t, NH);

        var (kw, vw) = kv.WriteView(li);
        Ops.SetRows(kw, k, pos);
        Ops.SetRows(vw, v, pos);
        var (kf, vf) = kv.ReadView(li, lk);

        var attn = Ops.FlashAttn(q, kf, vf, mask, 1f / MathF.Sqrt(HD));
        var flat = Ops.Reshape(Ops.Contiguous(attn), new long[] { t, D });
        x = Ops.Add(residual, Ops.Add(Ops.Matmul(flat, w[p + "attn.c_proj.weight"]), w[p + "attn.c_proj.bias"]));

        residual = x;
        var h2 = Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w[p + "ln_2.weight"]), w[p + "ln_2.bias"]);
        var fc = Ops.Gelu(Ops.Add(Ops.Matmul(h2, w[p + "mlp.c_fc.weight"]), w[p + "mlp.c_fc.bias"]));
        return Ops.Add(x, Ops.Add(Ops.Matmul(fc, w[p + "mlp.c_proj.weight"]), w[p + "mlp.c_proj.bias"]));
    }

    /// <summary>emb PT [t, D] -> final_norm output, all layers reading/writing the cache.</summary>
    public static Tensor ForwardCached(GgufWeights w, Tensor x, KvCache kv, Tensor pos, Tensor mask,
                                       int lk, int t)
    {
        for (int li = 0; li < NL; li++) x = LayerCached(w, li, x, kv, pos, mask, lk, t);
        var lastHidden = Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w["gpt.gpt.ln_f.weight"]),
                                 w["gpt.gpt.ln_f.bias"]);
        return Ops.Add(Ops.Mul(Ops.LayerNorm(lastHidden, 1e-5f), w["gpt.final_norm.weight"]),
                       w["gpt.final_norm.bias"]);
    }
}
