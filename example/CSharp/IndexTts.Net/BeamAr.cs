using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// HF <c>beam_sample</c> over N independent KV caches — the reference's
/// (num_beams=3, do_sample=True) generation path.
///
/// Each step: per-beam warped log-probs are combined with the beam's cumulative score into a
/// joint distribution over (beam, token); 2N candidates are sampled from it, the top N become
/// the next beams, and each destination beam's cache is reordered to its source's.
///
/// The reference samples with torch's RNG, which cannot be reproduced here, so this matches the
/// *structure* (warpers with min_tokens_to_keep=2, cumulative scoring, 2N sampling, reorder) and
/// not the exact draw.
/// </summary>
public sealed class BeamAr : IDisposable
{
    private const int V = Gpt2.MelVocab;

    private readonly Runtime _rt;
    private readonly Device _dev;
    private readonly int _n;
    private readonly ArDecoder[] _beam;
    private readonly KvCache _scratch;

    public BeamAr(Runtime rt, Device dev, GgufWeights w, int maxCtx, int numBeams)
    {
        _rt = rt; _dev = dev; _n = numBeams;
        _beam = new ArDecoder[numBeams];
        for (int i = 0; i < numBeams; i++) _beam[i] = new ArDecoder(rt, dev, w, maxCtx);
        _scratch = Gpt2Cached.NewCache(dev, maxCtx);
    }

    public bool UseReplay
    {
        set { foreach (var b in _beam) b.UseReplay = value; }
    }

    public int[] Generate(float[] conds, int[] textIds, int lang, int maxNew, Random rng,
                          float repPenalty, float topP, float temp, int topK)
    {
        // HF: min_tokens_to_keep is 2 whenever num_beams > 1.
        var smp = new Sampler
        {
            RepPenalty = repPenalty, TopP = topP, Temperature = temp, TopK = topK,
            MinTokensToKeep = 2,
        };

        var logits = new float[]?[_n];
        logits[0] = _beam[0].Prefill(conds, textIds, lang);
        int nPast = _beam[0].PromptLen;
        for (int i = 1; i < _n; i++)
        {
            CopyCache(_beam[0].Cache, _beam[i].Cache, nPast);
            logits[i] = (float[])logits[0]!.Clone();      // identical copies of the same prefix
        }

        var score = new double[_n];
        var gen = new List<int>[_n];
        var alive = new bool[_n];
        for (int i = 0; i < _n; i++)
        {
            score[i] = i == 0 ? 0.0 : double.NegativeInfinity;
            gen[i] = new List<int>();
            alive[i] = i == 0;
        }

        // Finished sequences, kept as (tokens, cumulative log-prob) hypotheses.
        var hypo = new List<(List<int> Toks, double Score)>();

        var joint = new float[_n * V];
        for (int step = 0; step < maxNew; step++)
        {
            for (int b = 0; b < _n; b++)
            {
                int row = b * V;
                if (!alive[b])
                {
                    for (int v = 0; v < V; v++) joint[row + v] = float.NegativeInfinity;
                    continue;
                }
                smp.Reset(GptAr.Penalized(gen[b]));
                var s = (float[])logits[b]!.Clone();
                smp.LogProbs(s);
                for (int v = 0; v < V; v++) joint[row + v] = (float)(score[b] + s[v]);
            }

            // HF beam_sample: sample 2N distinct (beam, token) pairs from the joint distribution,
            // then keep the N best.
            var cand = Sampler.Sample(joint, 2 * _n, rng);
            Array.Sort(cand, (a, b) => joint[b].CompareTo(joint[a]));
            if (cand.Length < _n) break;

            var src = new int[_n];
            var tok = new int[_n];
            bool allStop = true;
            for (int d = 0; d < _n; d++)
            {
                src[d] = cand[d] / V;
                tok[d] = cand[d] % V;
                if (tok[d] != GptAr.StopMel) allStop = false;
            }
            if (allStop) break;

            var oldGen = new List<int>[_n];
            for (int d = 0; d < _n; d++) oldGen[d] = new List<int>(gen[d]);

            if (!IsIdentity(src)) Reorder(src, nPast);

            for (int d = 0; d < _n; d++)
            {
                gen[d] = new List<int>(oldGen[src[d]]);   // copy: destinations sharing a source would alias
                score[d] = joint[cand[d]];
                if (tok[d] == GptAr.StopMel)
                {
                    // length_penalty is 0.0 in the reference, so the score is used as-is.
                    hypo.Add((new List<int>(gen[d]), score[d]));
                    alive[d] = false;
                    logits[d] = null;
                }
                else
                {
                    gen[d].Add(tok[d]);
                }
            }

            bool anyAlive = false;
            for (int d = 0; d < _n; d++)
                if (alive[d])
                {
                    logits[d] = _beam[d].Step(tok[d], nPast);
                    anyAlive = true;
                }
            nPast++;
            if (!anyAlive) break;
        }

        if (hypo.Count > 0)
        {
            var best = hypo[0];
            foreach (var h in hypo) if (h.Score > best.Score) best = h;
            return best.Toks.ToArray();
        }
        int b0 = 0;
        for (int i = 1; i < _n; i++) if (alive[i] && (!alive[b0] || score[i] > score[b0])) b0 = i;
        return gen[b0].ToArray();
    }

    // ---- cache reordering --------------------------------------------------------------------

    private static bool IsIdentity(int[] src)
    {
        for (int i = 0; i < src.Length; i++) if (src[i] != i) return false;
        return true;
    }

    /// <summary>Move cache[src[d]] into cache[d] for every d, through a single scratch cache.
    /// The permutation is applied cycle by cycle so no source is overwritten before it is read.</summary>
    private void Reorder(int[] src, int nPast)
    {
        var visited = new bool[_n];
        for (int start = 0; start < _n; start++)
        {
            if (visited[start] || src[start] == start) { visited[start] = true; continue; }
            var cyc = new List<int>();
            int c = start;
            while (!visited[c]) { visited[c] = true; cyc.Add(c); c = src[c]; }

            CopyCache(_beam[cyc[0]].Cache, _scratch, nPast);
            for (int i = 0; i < cyc.Count - 1; i++)
                CopyCache(_beam[cyc[i + 1]].Cache, _beam[cyc[i]].Cache, nPast);
            CopyCache(_scratch, _beam[cyc[^1]].Cache, nPast);
        }
    }

    /// <summary>Copy the first <paramref name="rows"/> KV slots of one cache into another.</summary>
    private void CopyCache(KvCache src, KvCache dst, int rows)
    {
        using var g = new Graph(_rt, _dev, 4096);
        g.Enter();
        for (int l = 0; l < src.Layers; l++)
        {
            Ops.Cpy(Ops.View3d(src.K, src.HeadDim, rows, src.Heads, src.Nb1, src.Nb2, src.LayerOffset(l)),
                    Ops.View3d(dst.K, dst.HeadDim, rows, dst.Heads, dst.Nb1, dst.Nb2, dst.LayerOffset(l)));
            Ops.Cpy(Ops.View3d(src.V, src.HeadDim, rows, src.Heads, src.Nb1, src.Nb2, src.LayerOffset(l)),
                    Ops.View3d(dst.V, dst.HeadDim, rows, dst.Heads, dst.Nb1, dst.Nb2, dst.LayerOffset(l)));
        }
        g.Exit();
        g.Compute();
    }

    public void Dispose()
    {
        foreach (var b in _beam) b.Dispose();
        _scratch.Dispose();
    }
}
