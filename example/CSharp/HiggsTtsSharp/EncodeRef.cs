using System;
using System.Linq;
using VulkanTorch;

namespace HiggsTts;

/// <summary>
/// Port of higgstts_py/model.py — the HiggsTTS `encode_ref` path:
/// reference audio -> RVQ codes [T, 8].
///
/// Feature extractor (7x Conv1d + GroupNorm + GELU) -> feature projection ->
/// WavLM encoder (positional conv embed + 12 transformer layers + mean) ->
/// semantic encoder -> DAC acoustic encoder -> concat/FC -> RVQ (8 codebooks).
/// </summary>
public static class EncodeRef
{
    private static readonly int[] FeStrides = { 5, 2, 2, 2, 2, 2, 2 };
    private static readonly int[] FeKernels = { 10, 3, 3, 3, 3, 2, 2 };

    private const int WavlmLayers = 12;
    private const int WavlmHeads = 12;
    private const int WavlmHd = 64;

    // pos_conv_embed (HubertPositionalConvEmbedding): grouped Conv1d(768,768,k=128,groups=16,pad=64)
    private const int PceK = 128;
    private const int PceCg = 48;
    private const int PceNg = 16;
    private const int PceC = 768;

    private static readonly int[] AcStrides = { 8, 5, 4, 2, 3 };
    private static readonly int[] AcPads = { 4, 3, 2, 1, 2 };

    private const int RvqN = 8;
    private const int Hop = 960;  // prod(AcStrides)

    // ---------------------------------------------------------------- PCE weight fusion

    /// <summary>
    /// Fuse the weight_norm (dim=2, i.e. over the K axis) into a single grouped-conv
    /// weight, matching the reference: F[(i + j*K), oc] = g[i]*W[oc,j,i]/norm[i].
    /// Stored as PT [OC, K*Cg].
    /// </summary>
    public static void BuildPceWeight(HiggsWeights w)
    {
        int k = PceK, cg = PceCg, c = PceC;
        var g = F16ToF32(w["codec.sem.encoder.pce.conv.pm.weight.orig0"].ReadBytes());
        var wflat = F16ToF32(w["codec.sem.encoder.pce.conv.pm.weight.orig1"].ReadBytes());

        var norm = new float[k];
        for (int i = 0; i < k; i++)
        {
            double s = 0;
            for (int oc = 0; oc < c; oc++)
                for (int j = 0; j < cg; j++)
                {
                    float v = wflat[(oc * cg + j) * k + i];
                    s += (double)v * v;
                }
            norm[i] = (float)Math.Sqrt(s);
        }

        var f = new float[(long)c * k * cg];  // PT [C, K*Cg]
        for (int oc = 0; oc < c; oc++)
            for (int a = 0; a < k * cg; a++)
            {
                int i = a % k, j = a / k;
                f[(long)oc * k * cg + a] = wflat[(oc * cg + j) * k + i] * (g[i] / norm[i]);
            }
        w.Put("__pce_fused", new long[] { c, (long)k * cg }, f);
    }

    private static float[] F16ToF32(byte[] bytes)
    {
        var f = new float[bytes.Length / 2];
        for (int i = 0; i < f.Length; i++)
            f[i] = (float)BitConverter.ToHalf(bytes, i * 2);
        return f;
    }

    // ---------------------------------------------------------------- pieces

    private static Tensor Pce(HiggsWeights w, Tensor x, int t)
    {
        var wf = w["__pce_fused"];                      // PT [768, 6144]
        const int kc = PceK * PceCg;
        const ulong nbb = PceC * 4;                     // nb1 of x
        Tensor? acc = null;
        for (int gi = 0; gi < PceNg; gi++)
        {
            var xg = Ops.Contiguous(Ops.View2d(x, PceCg, t, nbb, (ulong)(gi * PceCg * 4)));
            var cols = Ops.Im2colRafa(xg, PceK, 1, 64, 1);
            cols = Ops.View2d(cols, kc, t, (ulong)(kc * 4), 0);            // SamePad: crop last col
            var wg = Ops.View2d(wf, kc, PceCg, (ulong)(kc * 4), (ulong)((long)gi * PceCg * kc * 4));
            var o = Ops.MulMat(cols, wg);                                 // PT [48, T]
            acc = acc is null ? o : Ops.Concat(acc, o, 0);
        }
        var bias = Ops.Reshape(w["codec.sem.encoder.pce.conv.bias"], new long[] { PceC, 1 });
        acc = Ops.Gelu(Ops.Add(acc!, bias));
        return Ops.Transpose(acc);                                        // PT [T, 768]
    }

    private static Tensor NormOverTime(Tensor x)
        => Ops.Contiguous(Ops.Transpose(Ops.LayerNorm(Ops.Contiguous(Ops.Transpose(x)), 1e-5f)));

    private static Tensor LayernormAffine(Tensor x, Tensor wgt, Tensor bias, float eps = 1e-5f)
        => Ops.Add(Ops.Mul(Ops.LayerNorm(x, eps), wgt), bias);

    private static Tensor FeatureExtractor(HiggsWeights w, Tensor x)
    {
        for (int i = 0; i < 7; i++)
        {
            x = Ops.Conv1d(x, w[$"codec.sem.fe.cv.{i}.conv.weight"], FeStrides[i], 0, 1);
            if (i == 0)
            {
                x = NormOverTime(x);
                x = Ops.Add(Ops.Mul(x, w["codec.sem.fe.cv.0.ln.weight"]), w["codec.sem.fe.cv.0.ln.bias"]);
            }
            x = Ops.Gelu(x);
        }
        return x;
    }

    private static Tensor FeatureProjection(HiggsWeights w, Tensor x)
    {
        x = Ops.LayerNorm(x, 1e-5f);
        x = Ops.Add(Ops.Mul(x, w["codec.sem.fp.ln.weight"]), w["codec.sem.fp.ln.bias"]);
        return Ops.Add(Ops.Linear(x, w["codec.sem.fp.projection.weight"]),
                       w["codec.sem.fp.projection.bias"]);
    }

    private static Tensor ToHeads(Tensor x, int t)
    {
        var x4 = Ops.Reshape(x, new long[] { t, WavlmHeads, WavlmHd, 1 });
        var y4 = Ops.PermutePt(x4, 1, 0, 2, 3);
        return Ops.Reshape(Ops.Contiguous(y4), new long[] { WavlmHeads, t, WavlmHd });
    }

    private static Tensor WavlmAttention(HiggsWeights w, int li, Tensor x)
    {
        string p = $"codec.sem.enc.{li}.";
        int t = (int)x.Shape[0];
        Tensor Proj(string name) => Ops.Add(Ops.Linear(x, w[p + name + ".weight"]), w[p + name + ".bias"]);

        var q = ToHeads(Proj("attn_q"), t);
        var k = ToHeads(Proj("attn_k"), t);
        var v = ToHeads(Proj("attn_v"), t);

        float scale = 1f / MathF.Sqrt(WavlmHd);
        var attn = Ops.FlashAttn(q, k, v, null, scale);        // PT [T, nh, hd]
        attn = Ops.Reshape(Ops.Contiguous(attn), new long[] { t, WavlmHeads * WavlmHd });
        return Ops.Add(Ops.Linear(attn, w[p + "attn_out.weight"]), w[p + "attn_out.bias"]);
    }

    private static Tensor WavlmLayer(HiggsWeights w, int li, Tensor x)
    {
        string p = $"codec.sem.enc.{li}.";
        var h = Ops.Add(x, WavlmAttention(w, li, x));
        h = LayernormAffine(h, w[p + "ln.weight"], w[p + "ln.bias"]);
        var ffn = Ops.Gelu(Ops.Add(Ops.Linear(h, w[p + "ffn1.weight"]), w[p + "ffn1.bias"]));
        ffn = Ops.Add(Ops.Linear(ffn, w[p + "ffn2.weight"]), w[p + "ffn2.bias"]);
        var o = Ops.Add(h, ffn);
        return LayernormAffine(o, w[p + "fin_ln.weight"], w[p + "fin_ln.bias"]);
    }

    private static Tensor WavlmEncoder(HiggsWeights w, Tensor x)
    {
        int t = (int)x.Shape[0];
        x = Ops.Add(x, Pce(w, x, t));
        x = LayernormAffine(x, w["codec.sem.encoder.ln.weight"], w["codec.sem.encoder.ln.bias"]);
        var outs = new Tensor[WavlmLayers];
        var cur = x;
        for (int li = 0; li < WavlmLayers; li++)
        {
            cur = WavlmLayer(w, li, cur);
            outs[li] = cur;
        }
        var acc = outs[0];
        for (int i = 1; i < outs.Length; i++) acc = Ops.Add(acc, outs[i]);
        return Ops.Scale(acc, 1f / WavlmLayers);
    }

    private static Tensor AcousticEncoder(HiggsWeights w, Tensor x)
    {
        const string p = "codec.ac_enc.";
        x = Ops.Add(Ops.Conv1d(x, w[p + "conv1.weight"], 1, 3, 1), w[p + "conv1.bias"]);
        for (int bi = 0; bi < 5; bi++)
        {
            string b = p + $"block.{bi}.";
            foreach (var (ri, dil) in new[] { (1, 1), (2, 3), (3, 9) })
            {
                string r = b + $"res_unit{ri}.";
                var residual = x;
                var h = Ops.Snake1d(x, w[r + "snake1.alpha"]);
                long k1 = w[r + "conv1.weight"].Shape[2];
                h = Ops.Add(Ops.Conv1d(h, w[r + "conv1.weight"], 1, (int)((k1 - 1) / 2 * dil), dil),
                            w[r + "conv1.bias"]);
                h = Ops.Snake1d(h, w[r + "snake2.alpha"]);
                h = Ops.Add(Ops.Conv1d(h, w[r + "conv2.weight"], 1, 0, 1), w[r + "conv2.bias"]);
                x = Ops.Add(residual, h);
            }
            x = Ops.Snake1d(x, w[b + "snake1.alpha"]);
            x = Ops.Add(Ops.Conv1d(x, w[b + "conv1.weight"], AcStrides[bi], AcPads[bi], 1),
                        w[b + "conv1.bias"]);
        }
        x = Ops.Snake1d(x, w[p + "snake1.alpha"]);
        return Ops.Add(Ops.Conv1d(x, w[p + "conv2.weight"], 1, 1, 1), w[p + "conv2.bias"]);
    }

    private static Tensor SemanticEncoder(HiggsWeights w, Tensor x)
    {
        x = Ops.Conv1d(x, w["codec.enc_sem.conv.weight"], 1, 1, 1);
        for (int bi = 0; bi < 2; bi++)
        {
            string p = $"codec.enc_sem.blk.{bi}.";
            for (int ri = 0; ri < 2; ri++)
            {
                string rp = p + $"ru.{ri}.";
                var residual = x;
                var h = Ops.Elu(x);
                h = Ops.Conv1d(h, w[rp + "conv1.weight"], 1, 1, 1);
                h = Ops.Elu(h);
                h = Ops.Conv1d(h, w[rp + "conv2.weight"], 1, 0, 1);
                x = Ops.Add(residual, h);
            }
            x = Ops.Conv1d(x, w[p + "conv.weight"], 1, 1, 1);
            x = Ops.Add(x, w[p + "conv.bias"]);
        }
        return x;
    }

    /// <summary>8 residual quantizers; nearest neighbour via -||z-e||^2 = 2 z·e - ||z||² - ||e||².</summary>
    private static Tensor[] RvqEncode(HiggsWeights w, Tensor x)
    {
        var idsList = new Tensor[RvqN];
        var residual = x;
        for (int q = 0; q < RvqN; q++)
        {
            string p = $"codec.quant.{q}.";
            var cb = Ops.Cast(w[p + "codebook.embed"], Ops.F32);            // PT [cb_size, cb_dim]
            var z = Ops.Add(Ops.Linear(residual, w[p + "project_in.weight"]), w[p + "project_in.bias"]);
            var dot = Ops.Linear(z, cb);                                    // [T, cb_size]
            var x2 = Ops.SumRows(Ops.Mul(z, z));                            // [T, 1]
            var e2t = Ops.Transpose(Ops.SumRows(Ops.Mul(cb, cb)));          // [1, cb_size]
            var score = Ops.Sub(Ops.Sub(Ops.Scale(dot, 2f), x2), e2t);      // [T, cb_size]
            var ids = Ops.Argmax(score);                                    // [T, 1] I32
            idsList[q] = ids;
            var zq = Ops.GetRows(cb, ids);
            var pq = Ops.Add(Ops.Linear(zq, w[p + "project_out.weight"]), w[p + "project_out.bias"]);
            residual = Ops.Sub(residual, pq);
        }
        return idsList;
    }

    private static Tensor PrefillFused(HiggsWeights w, Tensor semInput, Tensor acIn)
    {
        var fp = FeatureProjection(w, FeatureExtractor(w, semInput));
        var se = SemanticEncoder(w, WavlmEncoder(w, fp));
        var ae = AcousticEncoder(w, acIn);
        var combined = Ops.Concat(ae, se, 1);
        return Ops.Add(Ops.Linear(combined, w["codec.fc.weight"]), w["codec.fc.bias"]);
    }

    // ---------------------------------------------------------------- public entry

    private static int FeLen(int l)
    {
        for (int i = 0; i < 7; i++) l = (l - FeKernels[i]) / FeStrides[i] + 1;
        return l;
    }

    private static int ConvLen(int l, int k, int s, int p) => (l + 2 * p - (k - 1) - 1) / s + 1;

    private static int AcLen(int l)
    {
        l = ConvLen(l, 7, 1, 3);
        int[] ks = { 16, 10, 8, 4, 6 };
        for (int i = 0; i < 5; i++) l = ConvLen(l, ks[i], AcStrides[i], AcPads[i]);
        return ConvLen(l, 3, 1, 1);
    }

    /// <summary>Reference audio -> RVQ codes PT [T, 8]. Mirrors tts.py HiggsTTS.encode_ref.</summary>
    public static int[,] Run(Runtime rt, Device dev, HiggsWeights w, float[] waveform, int sampleRate)
    {
        var wav = (float[])waveform.Clone();
        if (sampleRate != 24000) wav = Resampler.Resample(wav, sampleRate, 24000);

        var sem = Resampler.Resample(wav, 24000, 16000);
        var semPad = new float[sem.Length + 320];
        Array.Copy(sem, 0, semPad, 160, sem.Length);

        int tSem = FeLen(semPad.Length);
        int tAc = AcLen(wav.Length);
        int pad = tSem - tAc;
        float[] pcm;
        if (pad > 0)
        {
            int left = pad / 2 * Hop, right = (pad - pad / 2) * Hop;
            pcm = new float[wav.Length + left + right];
            Array.Copy(wav, 0, pcm, left, wav.Length);
        }
        else pcm = wav;

        using var g = new Graph(rt, dev);
        g.Enter();
        var xs = g.Input(new long[] { semPad.Length, 1 }, semPad);
        var xa = g.Input(new long[] { pcm.Length, 1 }, pcm);
        var fused = PrefillFused(w, xs, xa);
        var ids = RvqEncode(w, fused);

        var conts = ids.Select(i => Ops.Contiguous(i).MarkOutput()).ToArray();
        var flat = conts.Select(c => c.ToInts(g)).ToArray();   // first read computes the graph
        g.Exit();

        int t = flat[0].Length;
        var codes = new int[t, RvqN];
        for (int q = 0; q < RvqN; q++)
            for (int i = 0; i < t; i++)
                codes[i, q] = flat[q][i];
        return codes;
    }
}
