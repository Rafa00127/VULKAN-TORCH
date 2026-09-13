using VulkanTorch;

namespace IndexTts;

/// <summary>
/// Conditioning-embedding helpers from indextts/gpt/model_v2_5.py:
///   spk_emb_proj: campplus speaker style [192] -> [1280]
///   get_emovec / merge_emovec: raw 1024-dim conditioning latents -> [1280]
///     Conformer -> Perceiver -> emovec_layer -> emo_layer.
/// merge_emovec(a, b, alpha=1) == get_emovec(b) for alpha == 1.
/// </summary>
public static class EmoCond
{
    /// <summary>campplus style PT [192] -> spk conditioning PT [1, 1280].</summary>
    public static Tensor SpkEmbProj(GgufWeights w, Tensor spk)
        => Ops.Add(Ops.Linear(spk, w["gpt.spk_emb_proj.weight"]), w["gpt.spk_emb_proj.bias"]);

    /// <summary>Raw conditioning latents PT [Te, 1024] -> emotion vector PT [1, 1280].</summary>
    public static Tensor GetEmovec(Graph g, GgufWeights w, Tensor latent, int te)
    {
        var conf = Conformer.Forward(g, w, latent, te);       // PT [T', 512]
        int t = (int)conf.Shape[0];
        var perc = Perceiver.Forward(w, conf, t);          // PT [1, 1024]
        var v = Ops.Add(Ops.Linear(perc, w["gpt.emovec_layer.weight"]), w["gpt.emovec_layer.bias"]);
        return Ops.Add(Ops.Linear(v, w["gpt.emo_layer.weight"]), w["gpt.emo_layer.bias"]);
    }
}
