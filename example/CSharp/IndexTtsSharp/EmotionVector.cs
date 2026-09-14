using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// Emotion-vector control (the reference's <c>emo_vector</c> argument).
///
/// Eight emotion groups each own a block of conditioning rows. For each one the reference picks
/// the speaker row closest to the current voice by cosine similarity, then blends:
///
///   emovec = Σᵢ wᵢ · emo_matrix[i][argmax_j cos(style, spk_matrix[i][j])] + (1 − Σᵢwᵢ) · base
///
/// where <c>base</c> is <see cref="EmoCond.GetEmovec"/> of the speaker conditioning. The two
/// matrices ship inside the GGUF as <c>emo.spk_matrix</c> [73,192] and <c>emo.emo_matrix</c>
/// [73,1280] (converted from feat1.pt / feat2.pt).
/// </summary>
public static class EmotionVector
{
    /// <summary>Group order matters — it must match the reference's emo_num.</summary>
    public static readonly string[] Names =
        { "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm" };

    /// <summary>Rows per group; sums to 73.</summary>
    public static readonly int[] Counts = { 3, 17, 2, 8, 4, 5, 10, 24 };

    private const int StyleDim = 192, Dim = 1280;

    // normalize_emo_vec's user-experience bias: de-emphasises the emotions that tend to produce
    // strange results.
    private static readonly float[] Bias =
        { 0.9375f, 0.875f, 1f, 1f, 0.9375f, 0.9375f, 0.6875f, 0.5625f };

    /// <summary>Applies the per-emotion bias, then caps the total weight at 0.8.</summary>
    public static float[] Normalize(float[] weights)
    {
        var v = new float[Counts.Length];
        float sum = 0;
        for (int i = 0; i < v.Length; i++) { v[i] = weights[i] * Bias[i]; sum += v[i]; }
        if (sum > 0.8f)
        {
            float s = 0.8f / sum;
            for (int i = 0; i < v.Length; i++) v[i] *= s;
        }
        return v;
    }

    /// <summary>style: PT [192] from CAMPPlus; base: the speaker's own emovec [1280].</summary>
    public static float[] Blend(GgufWeights w, float[] style, float[] baseEmovec, float[] rawWeights)
    {
        var spk = w["emo.spk_matrix"].ReadFloats();     // [73, 192]
        var emo = w["emo.emo_matrix"].ReadFloats();     // [73, 1280]
        var wv = Normalize(rawWeights);

        var outp = new float[Dim];
        float used = 0;
        int off = 0;
        for (int e = 0; e < Counts.Length; e++)
        {
            if (wv[e] != 0f)
            {
                int best = 0;
                double bestSim = double.NegativeInfinity;
                for (int j = 0; j < Counts[e]; j++)
                {
                    double sim = Cosine(style, spk, (off + j) * StyleDim);
                    if (sim > bestSim) { bestSim = sim; best = j; }
                }
                int row = off + best;
                for (int d = 0; d < Dim; d++) outp[d] += wv[e] * emo[row * Dim + d];
                used += wv[e];
            }
            off += Counts[e];
        }
        float rest = 1f - used;
        for (int d = 0; d < Dim; d++) outp[d] += rest * baseEmovec[d];
        return outp;
    }

    private static double Cosine(float[] a, float[] m, int off)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < StyleDim; i++)
        {
            double x = a[i], y = m[off + i];
            dot += x * y; na += x * x; nb += y * y;
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    /// <summary>Parses "happy=0.6,calm=0.4" into the 8 weights (unknown names are an error).</summary>
    public static float[] Parse(string spec)
    {
        var w = new float[Counts.Length];
        foreach (var part in spec.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            var kv = part.Split('=', 2);
            if (kv.Length != 2) throw new ArgumentException($"expected name=weight, got '{part}'");
            int i = Array.FindIndex(Names, n => n.Equals(kv[0].Trim(), StringComparison.OrdinalIgnoreCase));
            if (i < 0) throw new ArgumentException($"unknown emotion '{kv[0].Trim()}' (one of: {string.Join(", ", Names)})");
            w[i] = float.Parse(kv[1].Trim(), System.Globalization.CultureInfo.InvariantCulture);
        }
        return w;
    }
}
