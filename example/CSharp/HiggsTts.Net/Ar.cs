using System;
using System.Collections.Generic;
using Minitorch;

namespace HiggsTts;

/// <summary>Port of higgstts_py/ar.py — Qwen3 backbone, KV cache, fused head, sampling.</summary>
public static class Ar
{
    public const int D = 2560;
    public const int NH = 32;
    public const int NKV = 8;
    public const int HD = 128;
    public const int NLayers = 36;
    public const float RopeTheta = 1e6f;
    public const float Ecl = 1e-6f;

    public const int NCb = 8;
    public const int CbVocab = 1026;
    public const int Boc = 1024;
    public const int Eoc = 1025;

    // ---- prefill embeddings ----
    // text embedding at text positions + summed fused-audio embedding at the (-100) audio slots.
    public static Tensor BuildPrefillEmbeds(Graph g, HiggsWeights w, int[] promptIds, int[] delayed,
                                            int lAudio)
    {
        int l = promptIds.Length;
        int preLen = l;
        for (int i = 0; i < l; i++)
            if (promptIds[i] == -100) { preLen = i; break; }

        var safe = new int[l];
        for (int i = 0; i < l; i++) safe[i] = promptIds[i] == -100 ? 0 : promptIds[i];
        var textEmb = Ops.GetRows(w["token_embd.weight"], g.InputI32(new long[] { l }, safe));

        Tensor? audioEmb = null;
        for (int c = 0; c < NCb; c++)
        {
            var idx = new int[lAudio];
            for (int i = 0; i < lAudio; i++) idx[i] = c * CbVocab + delayed[i * NCb + c];
            var e = Ops.GetRows(w["fused_embed.weight"], g.InputI32(new long[] { lAudio }, idx));
            audioEmb = audioEmb is null ? e : Ops.Add(audioEmb, e);
        }

        var mask = new float[l];
        for (int i = 0; i < l; i++) mask[i] = promptIds[i] == -100 ? 0f : 1f;
        var masked = Ops.Mul(textEmb, g.Input(new long[] { l, 1 }, mask));

        var zerosPre = new float[(long)preLen * D];
        var full = Ops.Concat(g.Input(new long[] { preLen, D }, zerosPre), audioEmb!, 0);
        int post = l - preLen - lAudio;
        if (post > 0)
            full = Ops.Concat(full, g.Input(new long[] { post, D }, new float[(long)post * D]), 0);
        return Ops.Add(masked, full);
    }

    // PT [T, n*hd] -> PT [n, T, hd] (ne=[hd,T,n]) — layout the KV cache / flash-attn want.
    private static Tensor ToHeadsPerm(Tensor x, int t, int n)
    {
        var x4 = Ops.Reshape(x, new long[] { t, n, HD, 1 });
        return Ops.Reshape(Ops.Contiguous(Ops.PermutePt(x4, 1, 0, 2, 3)), new long[] { n, t, HD });
    }

    /// <summary>One Qwen3 layer with KV-cache write/read. x: PT [T, D] -> PT [T, D].</summary>
    public static Tensor Layer(HiggsWeights w, int li, Tensor x, Tensor kvK, Tensor kvV, Tensor? pos,
                               Tensor? mask, int nPast, int maxCtx, int t)
    {
        string p = $"blk.{li}.";
        ulong nb1 = (ulong)(2 * HD), nb2 = (ulong)(2 * HD * maxCtx), nb3 = (ulong)(2 * HD * maxCtx * NKV);

        var residual = x;
        var h = Ops.Mul(Ops.RmsNorm(x, Ecl), w[p + "attn_norm.weight"]);

        var q3 = Ops.Reshape(Ops.Linear(h, w[p + "attn_q.weight"]), new long[] { t, NH, HD });
        var k3 = Ops.Reshape(Ops.Linear(h, w[p + "attn_k.weight"]), new long[] { t, NKV, HD });
        var v3 = Ops.Reshape(Ops.Linear(h, w[p + "attn_v.weight"]), new long[] { t, NKV, HD });
        q3 = Ops.Mul(Ops.RmsNorm(q3, Ecl), w[p + "attn_q_norm.weight"]);
        k3 = Ops.Mul(Ops.RmsNorm(k3, Ecl), w[p + "attn_k_norm.weight"]);

        q3 = Ops.Rope(q3, pos, HD, 2, 0, RopeTheta, 1f, 0f, 1f, 32f, 1f);
        k3 = Ops.Rope(k3, pos, HD, 2, 0, RopeTheta, 1f, 0f, 1f, 32f, 1f);
        var q = ToHeadsPerm(q3, t, NH);

        var kpo = ToHeadsPerm(k3, t, NKV);
        var vpo = ToHeadsPerm(v3, t, NKV);
        ulong off = (ulong)li * nb3 + (ulong)nPast * nb1;
        Ops.Cpy(kpo, Ops.View4d(kvK, HD, t, NKV, 1, nb1, nb2, nb3, off));
        Ops.Cpy(vpo, Ops.View4d(kvV, HD, t, NKV, 1, nb1, nb2, nb3, off));

        int lk = nPast + t;
        var kf = Ops.Contiguous(Ops.View3d(kvK, HD, lk, NKV, nb1, nb2, (ulong)li * nb3));
        var vf = Ops.Contiguous(Ops.View3d(kvV, HD, lk, NKV, nb1, nb2, (ulong)li * nb3));

        var attn = Ops.FlashAttn(q, kf, vf, mask, 1f / MathF.Sqrt(HD));
        attn = Ops.Reshape(Ops.Contiguous(attn), new long[] { t, NH * HD });
        x = Ops.Add(residual, Ops.Linear(attn, w[p + "attn_output.weight"]));

        residual = x;
        h = Ops.Mul(Ops.RmsNorm(x, Ecl), w[p + "ffn_norm.weight"]);
        var gate = Ops.Silu(Ops.Linear(h, w[p + "ffn_gate.weight"]));
        var up = Ops.Linear(h, w[p + "ffn_up.weight"]);
        x = Ops.Add(residual, Ops.Linear(Ops.Mul(gate, up), w[p + "ffn_down.weight"]));
        return x;
    }

    /// <summary>hidden_last: PT [1, D] (post output_norm) -> Tensor PT [N_CB, CB_VOCAB].</summary>
    public static Tensor FusedHeadLogits(HiggsWeights w, Tensor hiddenLast)
    {
        var lg = Ops.Linear(hiddenLast, w["fused_head.weight"]);
        return Ops.Reshape(lg, new long[] { NCb, CbVocab });
    }

    /// <summary>codes [N_CB] -> PT [1, D] (sum of per-codebook fused embeddings).</summary>
    public static Tensor EmbedCodes(Graph g, HiggsWeights w, int[] codes)
    {
        Tensor? e = null;
        for (int c = 0; c < NCb; c++)
        {
            var idx = new[] { c * CbVocab + codes[c] };
            var r = Ops.GetRows(w["fused_embed.weight"], g.InputI32(new long[] { 1 }, idx));
            e = e is null ? r : Ops.Add(e, r);
        }
        return e!;
    }

    /// <summary>Delayed codebook layout. codes: PT [T, N_CB] row-major.</summary>
    public static int[] ApplyDelayPattern(int[] codes, int t)
    {
        var outp = new int[(t + NCb - 1) * NCb];
        for (int i = 0; i < outp.Length; i++) outp[i] = Eoc;
        for (int c = 0; c < NCb; c++)
        {
            for (int i = 0; i < c; i++) outp[i * NCb + c] = Boc;
            for (int i = 0; i < t; i++) outp[(c + i) * NCb + c] = codes[i * NCb + c];
        }
        return outp;
    }

    public static int[] ReverseDelayPattern(int[] delayed, int rows)
    {
        int t = rows - (NCb - 1);
        var outp = new int[t * NCb];
        for (int c = 0; c < NCb; c++)
            for (int i = 0; i < t; i++) outp[i * NCb + c] = delayed[(c + i) * NCb + c];
        return outp;
    }

    // ---- sampling: top-k + temperature, per codebook ----
    public static int[] SampleCodes(float[] logits, float temperature, Random rng, int topk = 50)
    {
        var outp = new int[NCb];
        var order = new int[CbVocab];
        var sc = new float[CbVocab];
        for (int c = 0; c < NCb; c++)
        {
            for (int i = 0; i < CbVocab; i++)
            {
                order[i] = i;
                sc[i] = logits[c * CbVocab + i] / Math.Max(temperature, 1e-6f);
            }
            Array.Sort(sc, order);      // ascending score
            Array.Reverse(sc);          // descending score
            Array.Reverse(order);
            double max = sc[0], sum = 0;
            for (int i = 0; i < topk; i++) sum += Math.Exp(sc[i] - max);
            double r = rng.NextDouble() * sum, acc = 0;
            int chosen = order[topk - 1];
            for (int i = 0; i < topk; i++)
            {
                acc += Math.Exp(sc[i] - max);
                if (r <= acc) { chosen = order[i]; break; }
            }
            outp[c] = chosen;
        }
        return outp;
    }

    public sealed class KVCache : IDisposable
    {
        public Memory Mem { get; }
        public Tensor K { get; }
        public Tensor V { get; }
        public int MaxCtx { get; }

        public KVCache(Device dev, int maxCtx)
        {
            MaxCtx = maxCtx;
            Mem = new Memory(dev, (ulong)(2L * NLayers * NKV * maxCtx * HD * 2));
            K = Mem.Tensor(new long[] { NLayers, NKV, maxCtx, HD }, 1);
            V = Mem.Tensor(new long[] { NLayers, NKV, maxCtx, HD }, 1);
        }

        public void Dispose() => Mem.Dispose();
    }

    /// <summary>Run prefill and return the last-position logits flattened [N_CB * CB_VOCAB].</summary>
    public static float[] PrefillLogits(Runtime rt, Device dev, HiggsWeights w, KVCache kv,
                                        int[] promptIds, int[] delayed, int lAudio)
    {
        int l = promptIds.Length;
        using var g = new Graph(rt, dev);
        g.Enter();

        var x = BuildPrefillEmbeds(g, w, promptIds, delayed, lAudio);
        var posArr = new int[l];
        for (int i = 0; i < l; i++) posArr[i] = i;
        var pos = g.InputI32(new long[] { l }, posArr);

        var maskF = new float[(long)l * l];
        for (int q = 0; q < l; q++)
            for (int k = 0; k < l; k++) maskF[q * l + k] = k <= q ? 0f : float.NegativeInfinity;
        var mask = Ops.Cast(g.Input(new long[] { l, l }, maskF), 1);

        for (int li = 0; li < NLayers; li++)
            x = Layer(w, li, x, kv.K, kv.V, pos, mask, 0, kv.MaxCtx, l);

        var hn = Ops.Mul(Ops.RmsNorm(x, Ecl), w["output_norm.weight"]);
        var last = Ops.Reshape(Ops.View2d(Ops.Contiguous(hn), D, 1, D * 4, (ulong)((l - 1) * D * 4)),
                               new long[] { 1, D });
        var logits = Ops.Contiguous(FusedHeadLogits(w, last)).MarkOutput();
        var lg = logits.ToFloats(g);
        g.Exit();
        return lg;
    }

    /// <summary>One decode step for a given code vector; returns logits [N_CB * CB_VOCAB].</summary>
    public static float[] StepLogits(Runtime rt, Device dev, HiggsWeights w, KVCache kv, int[] codes,
                                     int nPast)
    {
        using var g = new Graph(rt, dev);
        g.Enter();
        var e = EmbedCodes(g, w, codes);
        var pos = g.InputI32(new long[] { 1 }, new[] { nPast });
        for (int li = 0; li < NLayers; li++)
            e = Layer(w, li, e, kv.K, kv.V, pos, null, nPast, kv.MaxCtx, 1);
        var hn = Ops.Mul(Ops.RmsNorm(e, Ecl), w["output_norm.weight"]);
        var logits = Ops.Contiguous(FusedHeadLogits(w, hn)).MarkOutput();
        var lg = logits.ToFloats(g);
        g.Exit();
        return lg;
    }

    /// <summary>Full AR generation; returns raw codes PT [T, N_CB] (int32).</summary>
    public static int[] Generate(Runtime rt, Device dev, HiggsWeights w, int[] promptIds,
                                 int[] refCodes, int refRows, out int steps, float temperature = 0.9f,
                                 int seed = 42, int maxSteps = 400, int topk = 50)
    {
        var delayed = ApplyDelayPattern(refCodes, refRows);
        int la = delayed.Length / NCb;
        int l = promptIds.Length;
        int maxCtx = l + maxSteps + 8;
        using var kv = new KVCache(dev, maxCtx);
        var rng = new Random(seed);

        var lg = PrefillLogits(rt, dev, w, kv, promptIds, delayed, la);
        int nPast = l;
        var all = new List<int>(maxSteps * NCb);
        int delayCount = 0, eocCountdown = -1;
        steps = 0;

        for (int step = 0; step < maxSteps; step++)
        {
            var cn = SampleCodes(lg, temperature, rng, topk);
            if (delayCount < NCb)
            {
                if (delayCount + 1 < NCb)
                    for (int c = delayCount + 1; c < NCb; c++) cn[c] = Boc;
                delayCount++;
            }
            else if (eocCountdown >= 0) eocCountdown--;
            else if (cn[0] == Eoc) eocCountdown = NCb - 2;

            all.AddRange(cn);
            steps++;
            if (eocCountdown == 0) break;

            lg = StepLogits(rt, dev, w, kv, cn, nPast);
            nPast++;
        }
        return ReverseDelayPattern(all.ToArray(), steps);
    }
}
