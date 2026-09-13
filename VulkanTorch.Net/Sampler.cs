using System;
using System.Collections.Generic;

namespace VulkanTorch;

/// <summary>
/// HuggingFace-equivalent logit processing for autoregressive sampling. The order
/// matches HF's `generate` pipeline:
///
///   1. repetition penalty on the RAW logits (HF's RepetitionPenaltyLogitsProcessor
///      runs before softmax; negative logits are multiplied, positive ones divided)
///   2. log-softmax
///   3. divide by temperature (applied to the log-probs -- equivalent to HF dividing
///      the logits, since the resulting *relative* probabilities are identical)
///   4. top-k
///   5. top-p (drop the low tail until the retained mass exceeds top_p)
///
/// The token set passed to <see cref="Reset"/> is what the repetition penalty applies
/// to (typically the prompt plus everything generated so far).
/// </summary>
public sealed class Sampler
{
    public float RepPenalty = 1f;
    public float TopP = 1f;
    public float Temperature = 1f;
    public int TopK = 0;
    public int MinTokensToKeep = 2;

    private readonly HashSet<int> _penalized = new();

    /// <summary>Tokens the repetition penalty applies to.</summary>
    public void Reset(IEnumerable<int> tokens)
    {
        _penalized.Clear();
        foreach (var t in tokens) _penalized.Add(t);
    }

    public void Add(int token) => _penalized.Add(token);

    /// <summary>In-place logit processing over <paramref name="s"/>.</summary>
    public void LogProbs(float[] s)
    {
        if (RepPenalty != 1f)
            foreach (var t in _penalized)
                if (t >= 0 && t < s.Length)
                    s[t] = s[t] < 0f ? s[t] * RepPenalty : s[t] / RepPenalty;

        float max = float.NegativeInfinity;
        foreach (var v in s) if (v > max) max = v;
        double total = 0;
        foreach (var v in s) total += Math.Exp(v - max);
        float logTotal = (float)Math.Log(total);
        for (int i = 0; i < s.Length; i++) s[i] = (s[i] - max - logTotal) / Temperature;

        if (TopK > 0 && TopK < s.Length)
        {
            float thr = KthLargest(s, Math.Max(TopK, MinTokensToKeep));
            for (int i = 0; i < s.Length; i++) if (s[i] < thr) s[i] = float.NegativeInfinity;
        }

        if (TopP > 0f && TopP < 1f)
        {
            var idx = new List<int>();
            float m = float.NegativeInfinity;
            for (int i = 0; i < s.Length; i++)
                if (float.IsFinite(s[i])) { idx.Add(i); if (s[i] > m) m = s[i]; }

            var w = new double[idx.Count];
            double tot = 0;
            for (int i = 0; i < idx.Count; i++) { w[i] = Math.Exp(s[idx[i]] - m); tot += w[i]; }

            // ascending by score so the low tail is removed first; ties broken by index
            var order = new int[idx.Count];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            Array.Sort(order, (a, b) =>
            {
                int c = s[idx[a]].CompareTo(s[idx[b]]);
                return c != 0 ? c : idx[a].CompareTo(idx[b]);
            });

            double cumulative = 0;
            double removeMass = 1.0 - TopP;
            int keepFrom = order.Length > MinTokensToKeep ? order.Length - MinTokensToKeep : 0;
            for (int i = 0; i < order.Length; i++)
            {
                cumulative += w[order[i]] / tot;
                if (i < keepFrom && cumulative <= removeMass)
                    s[idx[order[i]]] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>Normalised probabilities after <see cref="LogProbs"/>.</summary>
    public static double[] Probabilities(float[] logprobs)
    {
        var p = new double[logprobs.Length];
        double m = double.NegativeInfinity;
        foreach (var v in logprobs) if (v > m) m = v;
        double tot = 0;
        for (int i = 0; i < logprobs.Length; i++)
        {
            p[i] = float.IsFinite(logprobs[i]) ? Math.Exp(logprobs[i] - m) : 0.0;
            tot += p[i];
        }
        for (int i = 0; i < p.Length; i++) p[i] /= tot;
        return p;
    }

    /// <summary>
    /// Samples <paramref name="count"/> distinct token indices from the processed
    /// log-probs via the exponential race (Gumbel-max): rank = p_i / Exp(1)_i, take
    /// the top `count`. Distributionally exact, but not bit-identical to torch's RNG.
    /// </summary>
    public static int[] Sample(float[] logprobs, int count, Random rng)
    {
        float m = float.NegativeInfinity;
        foreach (var v in logprobs) if (v > m) m = v;

        var cand = new List<(int tok, double rank)>();
        for (int i = 0; i < logprobs.Length; i++)
        {
            if (!float.IsFinite(logprobs[i])) continue;
            double e = -Math.Log(1.0 - rng.NextDouble());
            cand.Add((i, Math.Exp(logprobs[i] - m) / e));
        }
        cand.Sort((a, b) => b.rank.CompareTo(a.rank));
        int n = Math.Min(count, cand.Count);
        var res = new int[n];
        for (int i = 0; i < n; i++) res[i] = cand[i].tok;
        return res;
    }

    private static float KthLargest(float[] s, int k)
    {
        var heap = new List<float>(k);   // min-heap of the k largest seen
        foreach (var v in s)
        {
            if (heap.Count < k) { heap.Add(v); heap.Sort(); }
            else if (v > heap[0]) { heap[0] = v; heap.Sort(); }
        }
        return heap.Count > 0 ? heap[0] : float.NegativeInfinity;
    }
}
