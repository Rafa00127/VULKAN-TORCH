using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 emotion perceiver: PerceiverResampler(dim=1024, dim_context=512,
/// ff_mult=2, heads=4, num_latents=1). Two cross-attention+FF blocks over a single
/// learnable latent, finished with an RMSNorm.
///
/// Faithful to indextts/gpt/perceiver.py. Note `cross_attn_include_queries=True`:
/// the key/value context is cat([latents, context]), so the latent is also its own
/// first key. The FF is `Linear → GEGLU → Linear` with inner dim int(1024*2*2/3)=1365.
/// </summary>
public static class Perceiver
{
    public const int D = 1024;
    public const int DimCtx = 512;
    public const int Heads = 4;
    public const int HeadDim = 64;
    public const int Depth = 2;
    private const int Inner = Heads * HeadDim;   // 256
    private const int FfInner = 1365;            // int(1024 * (2*2/3))
    private const string P = "gpt.emo_perceiver_encoder.";

    private static Tensor Ln(GgufWeights w, string p, Tensor x)
        => Ops.Add(Ops.Linear(x, w[p + ".weight"]), w[p + ".bias"]);

    /// <summary>context PT [T, DimCtx] -> latents PT [1, D].</summary>
    public static Tensor Forward(GgufWeights w, Tensor x, int t)
    {
        var ctx = Ln(w, P + "proj_context", x);        // PT [T, D]
        var latents = Ops.Cast(w[P + "latents"], Ops.F32);   // PT [1, D] (f16 in the GGUF)
        int nkv = t + 1;

        for (int li = 0; li < Depth; li++)
        {
            string q = $"{P}layers.{li}.";

            // ---- attention (context = cat([latents, ctx])) ----
            var qh = Attn.ToHeads(Ops.Linear(latents, w[q + "0.to_q.weight"]), 1, Heads, HeadDim);
            var kv = Ops.Linear(Ops.Concat(latents, ctx, 0), w[q + "0.to_kv.weight"]);  // PT [nkv, 2*Inner]
            ulong rb = (ulong)(2 * Inner * 4);
            var kh = Attn.ToHeads(Ops.Contiguous(Ops.View2d(kv, Inner, nkv, rb, 0)), nkv, Heads, HeadDim);
            var vh = Attn.ToHeads(Ops.Contiguous(Ops.View2d(kv, Inner, nkv, rb, (ulong)(Inner * 4))),
                                  nkv, Heads, HeadDim);
            var o = Attn.FromHeads(Attn.Sdpa(qh, kh, vh, 1f / MathF.Sqrt(HeadDim)), 1, Heads, HeadDim);
            latents = Ops.Add(Ops.Linear(o, w[q + "0.to_out.weight"]), latents);

            // ---- feed-forward: Linear -> GEGLU -> Linear ----
            var ff = Ln(w, q + "1.0", latents);                       // PT [1, 2*FfInner]
            var a = Ops.Contiguous(Ops.View2d(ff, FfInner, 1, (ulong)(2 * FfInner * 4), 0));
            var gate = Ops.Contiguous(Ops.View2d(ff, FfInner, 1, (ulong)(2 * FfInner * 4), (ulong)(FfInner * 4)));
            latents = Ops.Add(Ln(w, q + "1.2", Ops.Mul(a, Ops.Gelu(gate))), latents);
        }

        // RMSNorm: F.normalize(x) * sqrt(dim) * gamma == x / rms(x) * gamma
        return Ops.Mul(Ops.RmsNorm(latents, 1e-6f), w[P + "norm.gamma"]);
    }
}
