using System;
using System.Collections.Generic;
using System.Linq;
using IndexTts;
using VulkanTorch;

namespace IndexTtsSharp;

/// <summary>
/// IndexTTS 2.5 synthesis facade: reference audio → <see cref="ReferenceVoice"/>, then
/// text + voice → 24 kHz PCM. Self-contained (owns its Runtime / device / weights).
/// The model runs at 22.05 kHz; <see cref="Decode"/> resamples to 24 kHz on the way out.
/// Not thread-safe — serialize calls on one instance.
/// </summary>
public sealed class Tts : IDisposable
{
    public const int OutputSampleRate = 24000;
    private const int ModelSampleRate = 22050;

    private static readonly string[] Prefixes =
        { "gpt.", "codec.", "s2mel.", "w2v.", "campplus.", "bigvgan.", "emo." };

    private readonly Runtime _rt;
    private readonly Device _dev;
    private readonly GgufWeights _w;
    private IndexTokenizer? _tok;
    private TextFrontend? _front;
    private readonly LengthRegulator _lr;
    private readonly SemanticCodec _codec;
    private readonly Dit _dit;
    private readonly Cfm _cfm;
    private readonly BigVgan _big;
    private readonly float[] _w2vMean, _w2vStd;
    private bool _disposed;

    public Tts(string modelGgufPath, string? tiktokenPath = null)
    {
        _rt = new Runtime();
        _dev = _rt.Gpu();
        _w = new GgufWeights(modelGgufPath, _dev, Prefixes, 6UL << 30);
        _w2vMean = _w["w2v.stats_mean"].ReadFloats();
        _w2vStd = _w["w2v.stats_std"].ReadFloats();
        _lr = new LengthRegulator(_w);
        _codec = new SemanticCodec(_dev, _w);
        _dit = new Dit(_dev, _w);
        _cfm = new Cfm(_rt, _dev, _dit);
        _big = new BigVgan(_dev, _w);
        if (tiktokenPath != null) SetTokenizer(tiktokenPath);
    }

    public string? BackendName => _rt.Name;
    public int TensorCount => _w.Count;

    /// <summary>Set / replace the tiktoken vocab (required before <see cref="GenerateCodes"/>).</summary>
    public bool SetTokenizer(string tiktokenPath)
    {
        _tok = new IndexTokenizer(tiktokenPath);
        _front = new TextFrontend(_tok);
        return true;
    }

    public bool HasTokenizer => _front != null;

    /// <summary>Reference audio (any sample rate) → a reusable <see cref="ReferenceVoice"/>.</summary>
    public ReferenceVoice EncodeReference(float[] audio, int sampleRate)
    {
        var wav = (float[])audio.Clone();
        NormalizeRefLevel(wav);
        var w16 = Resampler.Resample(wav, sampleRate, 16000);
        var w22 = Resampler.Resample(wav, sampleRate, 22050);

        var fb = Dsp.KaldiFbank(ToDouble(w16), 16000, 80);
        int nf = fb.GetLength(0);
        float[] style = Style(fb, nf);

        var feats = Dsp.Stack((float[,])fb.Clone(), out var fmask);
        float[] spk = SpeakerEmbedding(feats, fmask);

        var refMel = Dsp.MelSpectrogram(ToDouble(w22));      // [80, T]
        int refFrames = refMel.GetLength(1);
        float[] promptCond = LengthReg(_lr, spk, refFrames);
        return new ReferenceVoice(style, spk, FlattenT(refMel), promptCond);
    }

    /// <summary>Text + voice → AR mel codes (segments joined by a <c>-1</c> sentinel).</summary>
    public int[] GenerateCodes(string text, ReferenceVoice v, IndexOptions? options = null)
    {
        options ??= new IndexOptions();
        var front = _front ?? throw new InvalidOperationException("no tokenizer; call SetTokenizer first");
        var enc = front.EncodeForInference(text, TextFrontend.DefaultMaxTokensPerSegment, options.Lang);

        float[] baseEmo = EmotionVec(v.Spk);
        float[] emovec = options.Emotion == null
            ? baseEmo
            : EmotionVector.Blend(_w, v.Style, baseEmo, EmotionVector.Parse(options.Emotion));
        float[] conds = CondPrefix(emovec);

        int maxSteps = options.MaxSteps > 0 ? options.MaxSteps : Math.Max(64, text.Length * 8);
        var rng = new Random(options.Seed);
        var all = new List<int>();
        for (int s = 0; s < enc.Segments.Count; s++)
        {
            if (s > 0) all.Add(-1);                 // segment boundary sentinel
            int langId = IndexTokenizer.LangToToken(enc.Lang);
            int[] codes;
            if (options.NumBeams > 1)
            {
                using var beam = new BeamAr(_rt, _dev, _w, 2048, options.NumBeams) { UseReplay = options.UseReplay };
                codes = beam.Generate(conds, enc.SegmentTokenIds[s], langId, maxSteps, rng,
                                      options.RepPenalty, options.TopP, options.Temperature, options.TopK);
            }
            else
            {
                using var ar = new ArDecoder(_rt, _dev, _w, 2048) { UseReplay = options.UseReplay };
                var logits = ar.Prefill(conds, enc.SegmentTokenIds[s], langId);
                codes = ar.Generate(logits, ar.PromptLen, maxSteps, rng,
                                    options.RepPenalty, options.TopP, options.Temperature, options.TopK);
            }
            all.AddRange(codes);
        }
        return all.ToArray();
    }

    /// <summary>AR mel codes + voice → 24 kHz float32 PCM.</summary>
    public float[] Decode(int[] codes, ReferenceVoice v, IndexOptions? options = null)
    {
        options ??= new IndexOptions();
        var rng = new Random(options.Seed);
        var pieces = new List<float[]>();
        foreach (var seg in SplitSegments(codes))
        {
            var sinfer = CodecDecode(_codec, seg);
            int targetLen = (int)(seg.Length * 2 * 1.72 * options.DurationFactor);   // codec doubles the time axis
            var cond = LengthReg(_lr, sinfer, targetLen);
            var catCond = new float[(v.RefFrames + targetLen) * 512];
            Array.Copy(v.PromptCond, catCond, v.PromptCond.Length);
            Array.Copy(cond, 0, catCond, v.PromptCond.Length, cond.Length);

            var z = new float[(v.RefFrames + targetLen) * 80];
            for (int i = 0; i < z.Length; i++) z[i] = (float)Gauss(rng);

            var vcFull = _cfm.Inference(z, v.RefMel, v.RefFrames, catCond, v.Style,
                                        options.CfgRate, options.DiffusionSteps);
            var vc = new float[targetLen * 80];
            Array.Copy(vcFull, v.RefFrames * 80, vc, 0, vc.Length);
            pieces.Add(BigVganRun(_big, vc, targetLen));
        }
        var all = Concat(pieces);
        return Resampler.Resample(all, ModelSampleRate, OutputSampleRate);
    }

    /// <summary>Text + voice → 24 kHz float32 PCM.</summary>
    public float[] Synthesize(string text, ReferenceVoice v, IndexOptions? options = null)
        => Decode(GenerateCodes(text, v, options), v, options);

    // ---- glue (lifted from the CLI) --------------------------------------------------------

    private static int[][] SplitSegments(int[] codes)
    {
        var segs = new List<int[]>();
        int start = 0;
        for (int i = 0; i <= codes.Length; i++)
            if (i == codes.Length || codes[i] == -1)
            {
                if (i > start) segs.Add(codes[start..i]);
                start = i + 1;
            }
        return segs.ToArray();
    }

    /// <summary>Normalise the reference waveform's RMS to a fixed target (peak-capped at 0.99).
    /// IndexTTS's conditioning amplifies reference loudness ~2.5x, so a loud reference drives
    /// the output to full scale. RMS (not peak) is the right metric — the refs differ in crest
    /// factor, not peak. HiggsTTS tracks its reference ~1:1 and needs no such fix.</summary>
    private static void NormalizeRefLevel(float[] x)
    {
        const float targetRms = 0.08f;      // ≈ -22 dBFS
        double s = 0, peak = 0;
        for (int i = 0; i < x.Length; i++) { s += (double)x[i] * x[i]; peak = Math.Max(peak, Math.Abs(x[i])); }
        double rms = Math.Sqrt(s / x.Length);
        if (rms < 1e-6 || peak < 1e-6) return;
        double gain = targetRms / rms;
        if (peak * gain > 0.99) gain = 0.99 / peak;
        for (int i = 0; i < x.Length; i++) x[i] *= (float)gain;
    }

    private float[] Style(float[,] fb, int nf)
    {
        var f = (float[,])fb.Clone();
        Dsp.SubtractColumnMean(f, nf, fb.GetLength(1));
        using var cp = new CampPlus(_dev, _w);
        using var g = new Graph(_rt, _dev, 300000);
        g.Enter();
        var t = cp.Forward(g, g.Input(new long[] { nf, fb.GetLength(1) }, Flat(f))).MarkOutput();
        var r = t.ToFloats(g);
        g.Exit();
        return r;
    }

    private float[] SpeakerEmbedding(float[,] feats, float[] mask)
    {
        int t = feats.GetLength(0);
        var model = new W2vBert(_w);
        using var g = new Graph(_rt, _dev, 300000);
        g.Enter();
        var h = model.Forward(g, g.Input(new long[] { t, feats.GetLength(1) }, Flat(feats)), mask,
                              _w2vMean, _w2vStd).MarkOutput();
        var r = h.ToFloats(g);
        g.Exit();
        return r;
    }

    private float[] CodecDecode(SemanticCodec codec, int[] codes)
    {
        using var g = new Graph(_rt, _dev, 200000);
        g.Enter();
        var t = codec.Decode(g, g.InputI32(new long[] { codes.Length }, codes)).MarkOutput();
        var r = t.ToFloats(g);
        g.Exit();
        return r;
    }

    private float[] LengthReg(LengthRegulator lr, float[] x, int ylens)
    {
        int t = x.Length / 1024;
        using var g = new Graph(_rt, _dev, 200000);
        g.Enter();
        var o = lr.Forward(g, g.Input(new long[] { t, 1024 }, x), ylens).MarkOutput();
        var r = o.ToFloats(g);
        g.Exit();
        return r;
    }

    private float[] EmotionVec(float[] spk)
    {
        int t = spk.Length / 1024;
        using var g = new Graph(_rt, _dev, 200000);
        g.Enter();
        // No separate emotion prompt: the emotion latent is the speaker conditioning itself,
        // so merge_emovec(spk, spk) collapses to get_emovec(spk).
        var e = EmoCond.GetEmovec(g, _w, g.Input(new long[] { t, 1024 }, spk), t).MarkOutput();
        var r = e.ToFloats(g);
        g.Exit();
        return r;
    }

    private static float[] CondPrefix(float[] emovec)
    {
        var c = new float[3 * 1280];
        Array.Copy(emovec, c, emovec.Length);       // rows 1-2 stay zero
        return c;
    }

    private float[] BigVganRun(BigVgan big, float[] vc, int vcLen)
    {
        using var g = new Graph(_rt, _dev, 300000);
        g.Enter();
        var o = big.Forward(g, g.Input(new long[] { vcLen, 80 }, vc)).MarkOutput();
        var r = o.ToFloats(g);
        g.Exit();
        return r;
    }

    private static double[] ToDouble(float[] x)
    {
        var d = new double[x.Length];
        for (int i = 0; i < x.Length; i++) d[i] = x[i];
        return d;
    }

    private static float[] Flat(float[,] a)
    {
        var r = new float[a.Length];
        Buffer.BlockCopy(a, 0, r, 0, a.Length * 4);
        return r;
    }

    /// <summary>mel is [80, T]; the rest of the pipeline wants [T, 80].</summary>
    private static float[] FlattenT(float[,] mel)
    {
        int c = mel.GetLength(0), t = mel.GetLength(1);
        var r = new float[t * c];
        for (int i = 0; i < t; i++)
            for (int j = 0; j < c; j++) r[i * c + j] = mel[j, i];
        return r;
    }

    private static double Gauss(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static float[] Concat(List<float[]> parts)
    {
        int gap = (int)(0.2 * ModelSampleRate);     // 0.2 s between segments
        int n = parts.Sum(p => p.Length) + Math.Max(0, parts.Count - 1) * gap;
        var r = new float[n];
        int at = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            Array.Copy(parts[i], 0, r, at, parts[i].Length);
            at += parts[i].Length;
            if (i < parts.Count - 1) at += gap;
        }
        return r;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _big.Dispose();
        _dit.Dispose();
        _codec.Dispose();
        _w.Dispose();
        _rt.Dispose();
        _disposed = true;
    }
}
