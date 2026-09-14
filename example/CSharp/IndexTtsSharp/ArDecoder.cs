using System;
using System.Collections.Generic;
using System.Linq;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 autoregressive mel-code generation over the shared <see cref="KvCache"/>.
///
/// Prefill writes KV rows 0..L-1; each decode step writes one row and replays a captured
/// T=1 graph whose window size is pinned to a bucket (see <see cref="GraphCache"/>), so
/// nothing in the graph depends on the position and it is captured once per bucket.
///
/// Sampling uses the shared HF-equivalent <see cref="Sampler"/>. This is the
/// single-sequence path (num_beams = 1); the reference model runs beam_sample with
/// num_beams = 3, which additionally needs per-beam cache reordering.
/// </summary>
public sealed class ArDecoder : IDisposable
{
    private readonly Runtime _rt;
    private readonly Device _dev;
    private readonly GgufWeights _w;
    private readonly KvCache _kv;
    private readonly BucketedReplay<StepInputs> _replay;

    /// <summary>false rebuilds the step graph every step — the graph cache's A/B lever.</summary>
    public bool UseReplay { get => _replay.Replay; set => _replay.Replay = value; }

    /// <summary>The KV cache this beam writes into (exposed for beam-search reordering).</summary>
    public KvCache Cache => _kv;

    public int MelLen { get; private set; }
    public int PromptLen { get; private set; }

    public ArDecoder(Runtime rt, Device dev, GgufWeights w, int maxCtx)
    {
        _rt = rt; _dev = dev; _w = w;
        _kv = Gpt2Cached.NewCache(dev, maxCtx);
        _replay = new BucketedReplay<StepInputs>(BuildStep);
    }

    /// <summary>Per-step inputs of the captured decode graph.</summary>
    private sealed class StepInputs
    {
        public Tensor Ids = null!;       // I32 [1]  mel token
        public Tensor PosIdx = null!;    // I32 [1]  mel position index
        public Tensor Pos = null!;       // I32 [1]  KV write position
        public Tensor MaskIn = null!;    // F32 [1, lk]
    }

    /// <summary>Write KV rows 0..L-1 and return the last-position logits [mel vocab].</summary>
    public float[] Prefill(float[] conds, int[] textIds, int lang)
    {
        using var g = new Graph(_rt, _dev);
        g.Enter();
        var condsT = g.Input(new long[] { GptAr.CondRows, Gpt2.D }, conds);
        var embeds = GptAr.PrepareGptInputs(g, _w, condsT, textIds, lang, out int melLen);
        MelLen = melLen;
        var prefill = Ops.Concat(embeds, GptAr.MelToken(g, _w, GptAr.StartMel, 0), 0);
        int t = (int)prefill.Shape[0];
        PromptLen = t;
        int lk = GraphCache.PickBucket(t);
        if (lk == 0) throw new InvalidOperationException($"prompt {t} exceeds the largest KV bucket");

        var pos = g.InputI32(new long[] { t }, Enumerable.Range(0, t).ToArray());
        var mask = Ops.Cast(g.Input(new long[] { t, lk }, GraphCache.CausalMask(t, lk, 0)), Ops.F16);
        var fn = Gpt2Cached.ForwardCached(_w, prefill, _kv, pos, mask, lk, t);
        var lg = Ops.Contiguous(Gpt2.MelHead(_w, fn)).MarkOutput();
        var all = lg.ToFloats(g);
        g.Exit();

        var last = new float[Gpt2.MelVocab];
        Array.Copy(all, (t - 1) * Gpt2.MelVocab, last, 0, Gpt2.MelVocab);
        return last;
    }

    /// <summary>One decode step: writes KV at <paramref name="nPast"/>, returns logits [mel vocab].</summary>
    public float[] Step(int token, int nPast)
        => _replay.Run(nPast, s =>
        {
            s.Set(s.In.Ids, new[] { token });
            // the reference's mel position index is (attn_len - mel_len), i.e. it skips 1
            s.Set(s.In.PosIdx, new[] { (nPast + 1) - MelLen });
            s.Set(s.In.Pos, new[] { nPast });
            s.Set(s.In.MaskIn, GraphCache.CausalMask(1, s.Window, nPast));
        });

    private StepGraph<StepInputs> BuildStep(int lk)
    {
        var g = new Graph(_rt, _dev);
        g.Enter();
        var e = new StepInputs();
        e.Ids = g.InputI32(new long[] { 1 }, new int[1]);
        e.PosIdx = g.InputI32(new long[] { 1 }, new int[1]);
        e.Pos = g.InputI32(new long[] { 1 }, new int[1]);
        e.MaskIn = g.Input(new long[] { 1, lk }, new float[lk]);
        var mask = Ops.Cast(e.MaskIn, Ops.F16);

        var emb = Ops.Add(Ops.GetRows(_w["gpt.mel_embedding.weight"], e.Ids),
                          Ops.GetRows(_w["gpt.mel_pos_embedding.emb.weight"], e.PosIdx));
        var fn = Gpt2Cached.ForwardCached(_w, emb, _kv, e.Pos, mask, lk, 1);
        var logits = Ops.Contiguous(Gpt2.MelHead(_w, fn)).MarkOutput();
        g.Exit();
        return new StepGraph<StepInputs>(g, e, logits, lk);
    }

    /// <summary>
    /// Sample mel codes until the stop token or <paramref name="maxNew"/> steps.
    /// The four warpers match the official config; min_tokens_to_keep is 1 here because
    /// this is the single-sequence path (the reference's num_beams=3 uses 2).
    /// </summary>
    public int[] Generate(float[] curLogits, int nextPast, int maxNew, Random rng,
                          float repPenalty = 10f, float topP = 0.8f, float temp = 0.8f, int topK = 30,
                          Action<int>? onStep = null)
    {
        var smp = new Sampler
        {
            RepPenalty = repPenalty, TopP = topP, Temperature = temp, TopK = topK,
            MinTokensToKeep = 1,
        };
        var codes = new List<int>();
        var cur = curLogits;
        int nPast = nextPast;
        for (int i = 0; i < maxNew; i++)
        {
            smp.Reset(GptAr.Penalized(codes));
            var s = (float[])cur.Clone();
            smp.LogProbs(s);
            int tok = Sampler.Sample(s, 1, rng)[0];
            if (tok == GptAr.StopMel) break;
            codes.Add(tok);
            onStep?.Invoke(codes.Count);
            cur = Step(tok, nPast);
            nPast++;
        }
        return codes.ToArray();
    }

    public void Dispose()
    {
        _replay.Dispose();
        _kv.Dispose();
    }
}
