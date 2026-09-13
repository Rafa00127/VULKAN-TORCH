using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// GPT autoregressive input assembly for IndexTTS 2.5 (indextts/gpt/model_v2_5.py).
///
/// The conditioning is a "soft prefix": cat([spk + emovec, zeros x 2]) PT [3, D],
/// followed by the text embeddings. Generation then appends one mel embedding per
/// step. Note GPT2InferenceModel names its params misleadingly — `text_pos_emb` is
/// the *mel* position embedding and `embeddings` is the *mel* embedding.
/// </summary>
public static class GptAr
{
    public const int StartText = 0;
    public const int StopText = 1;
    public const int StartMel = 8192;
    public const int StopMel = 8193;
    public const int CondRows = 3;

    private static Tensor EmbedRows(Graph g, Tensor weight, int[] ids)
        => Ops.GetRows(weight, g.InputI32(new long[] { ids.Length }, ids));

    /// <summary>conds PT [3, D] + text ids -> inputs_embeds PT [melLen, D].</summary>
    public static Tensor PrepareGptInputs(Graph g, GgufWeights w, Tensor conds, int[] textIds,
                                          int lang, out int melLen)
    {
        // text is wrapped with start/stop tokens, then embedded with positions and lang
        var ids = new int[textIds.Length + 2];
        ids[0] = StartText;
        Array.Copy(textIds, 0, ids, 1, textIds.Length);
        ids[^1] = StopText;
        var pos = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) pos[i] = i;

        var te = EmbedRows(g, w["gpt.text_embedding.weight"], ids);
        te = Ops.Add(te, EmbedRows(g, w["gpt.text_pos_embedding.emb.weight"], pos));
        te = Ops.Add(te, EmbedRows(g, w["gpt.lang_embedding.weight"], new[] { lang }));

        var emb = Ops.Concat(conds, te, 0);
        melLen = (int)emb.Shape[0];
        return emb;
    }

    /// <summary>mel token id + mel position index -> PT [1, D].</summary>
    public static Tensor MelToken(Graph g, GgufWeights w, int token, int pos)
        => Ops.Add(EmbedRows(g, w["gpt.mel_embedding.weight"], new[] { token }),
                   EmbedRows(g, w["gpt.mel_pos_embedding.emb.weight"], new[] { pos }));

    /// <summary>
    /// Tokens HF's repetition penalty applies to: the whole `input_ids`, which here is the
    /// all-ones prompt (stop_text) plus the start_mel token, plus everything generated.
    /// </summary>
    public static int[] Penalized(System.Collections.Generic.IEnumerable<int> generated)
    {
        var l = new System.Collections.Generic.List<int> { StopText, StartMel };
        l.AddRange(generated);
        return l.ToArray();
    }

    /// <summary>Full (uncached) forward over `emb` PT [t, D] -> mel logits PT [t, 8194].</summary>
    public static Tensor Logits(GgufWeights w, Graph g, Tensor emb, int t)
    {
        var (_, finalNorm) = Gpt2.Forward(w, g, emb, t);
        return Gpt2.MelHead(w, finalNorm);
    }
}
