using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 command line: reference wav + text -> wav, mirroring indextts/infer_v2_5.py.
///
///   synth (default)   --ref-wav r.wav --text "..." -> --out y.wav
///
/// Reference-audio chain: Kaldi fbank -> CAMPPlus (style); SeamlessM4T features -> Wav2Vec2Bert
/// (speaker conditioning); 22.05 kHz mel (ref_mel). Then per text segment: GPT-2 AR -> semantic
/// codec -> length regulator -> CFM (25-step Euler, CFG) -> BigVGAN.
/// </summary>
internal static class Cli
{
    private static Runtime _rt = null!;

    private static readonly string[] Prefixes =
        { "gpt.", "codec.", "s2mel.", "w2v.", "campplus.", "bigvgan.", "emo." };

    public static int Run(string[] args)
    {
        string? model = null, tokenizer = null, refWav = null, text = null, outPath = null, emoSpec = null;
        string lang = "zh";
        int maxSteps = 0, seed = 42, topK = 30, diffusionSteps = 25, numBeams = 1;
        float temp = 0.8f, topP = 0.8f, repPen = 10f, cfgRate = 0.7f, durationFactor = 1.0f;
        bool useReplay = true;

        for (int i = 1; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            switch (args[i])
            {
                case "--model": model = Next(); break;
                case "--tokenizer": tokenizer = Next(); break;
                case "--ref-wav": refWav = Next(); break;
                case "--text": text = Next(); break;
                case "--lang": lang = Next(); break;
                case "--emo": emoSpec = Next(); break;
                case "--out": outPath = Next(); break;
                case "--max-steps": maxSteps = int.Parse(Next()); break;
                // The reference defaults to 3, but HF scores beam hypotheses by the raw sum of
                // log-probs (length_penalty=0), so a beam that ends early tends to win. That makes
                // --num-beams 3 prone to truncated output here; 1 is the stable default.
                case "--num-beams": numBeams = int.Parse(Next()); break;
                case "--seed": seed = int.Parse(Next()); break;
                case "--top-k": topK = int.Parse(Next()); break;
                case "--top-p": topP = float.Parse(Next()); break;
                case "--temperature": temp = float.Parse(Next()); break;
                case "--rep-penalty": repPen = float.Parse(Next()); break;
                case "--cfg-rate": cfgRate = float.Parse(Next()); break;
                case "--diffusion-steps": diffusionSteps = int.Parse(Next()); break;
                case "--duration-factor": durationFactor = float.Parse(Next()); break;
                case "--no-graph-cache": useReplay = false; break;
                default: Console.Error.WriteLine($"unknown arg {args[i]}"); return 2;
            }
        }

        string root = Program.FindRoot();
        model ??= Path.Combine(root, "model", "indextts2.5", "indextts2.5.f16.gguf");
        tokenizer ??= Path.Combine(root, "model", "indextts2.5", "multilingual_zh_ja_yue_char_del.tiktoken");
        refWav ??= Path.Combine(root, "data", "ref_audio", "melinaref_24k.wav");
        text ??= "大家好，这是一个测试。";
        outPath ??= Path.Combine(root, "data", "indextts", "cli.wav");
        if (maxSteps <= 0) maxSteps = Math.Max(64, text.Length * 8);

        using var rt = new Runtime();
        _rt = rt;
        var dev = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");
        var sw = Stopwatch.StartNew();

        using var w = new GgufWeights(model, dev, Prefixes, 6UL << 30);
        Console.WriteLine($"weights: {w.Count} tensors in {sw.ElapsedMilliseconds} ms");

        // ---- text front-end ---------------------------------------------------------------
        var tok = new IndexTokenizer(tokenizer);
        var front = new TextFrontend(tok);
        var enc = front.EncodeForInference(text, TextFrontend.DefaultMaxTokensPerSegment, lang);
        Console.WriteLine($"segments: {enc.Segments.Count}  lang={enc.Lang}");

        // ---- reference audio --------------------------------------------------------------
        var wav = LoadWav(refWav, out int sr);
        var w16 = Resampler.Resample(wav, sr, 16000);
        var w22 = Resampler.Resample(wav, sr, 22050);
        var fb = Dsp.KaldiFbank(ToDouble(w16), 16000, 80);
        int nf = fb.GetLength(0);

        float[] style = Style(w, dev, fb, nf);

        var feats = Dsp.Stack((float[,])fb.Clone(), out var fmask);
        var (mean, std) = LoadW2vStats(w);
        float[] spk = SpeakerEmbedding(w, dev, feats, fmask, mean, std);
        Console.WriteLine($"ref: {nf} fbank frames, {feats.GetLength(0)} w2v frames");

        var refMel = Dsp.MelSpectrogram(ToDouble(w22));      // [80, T]
        int refMelLen = refMel.GetLength(1);
        Console.WriteLine($"ref_mel: 80x{refMelLen}   ({sw.ElapsedMilliseconds} ms total)");

        // ---- per-segment synthesis --------------------------------------------------------
        var lr = new LengthRegulator(w);
        var codec = new SemanticCodec(dev, w);
        var dit = new Dit(dev, w);
        var cfm = new Cfm(rt, dev, dit);
        var big = new BigVgan(dev, w);

        float[] promptCond = LengthReg(w, dev, lr, spk, refMelLen);
        float[] baseEmo = EmotionVec(w, dev, spk);
        float[] emovec = emoSpec == null ? baseEmo : EmotionVector.Blend(w, style, baseEmo, EmotionVector.Parse(emoSpec));
        if (emoSpec != null) Console.WriteLine($"emotion: {emoSpec}");
        float[] conds = CondPrefix(emovec);

        var rng = new Random(seed);
        var pieces = new List<float[]>();
        for (int s = 0; s < enc.Segments.Count; s++)
        {
            var t0 = Stopwatch.StartNew();
            int langId = IndexTokenizer.LangToToken(enc.Lang);
            int[] codes;
            if (numBeams > 1)
            {
                using var beam = new BeamAr(rt, dev, w, 2048, numBeams) { UseReplay = useReplay };
                codes = beam.Generate(conds, enc.SegmentTokenIds[s], langId, maxSteps, rng,
                                      repPen, topP, temp, topK);
            }
            else
            {
                using var ar = new ArDecoder(rt, dev, w, 2048) { UseReplay = useReplay };
                var logits = ar.Prefill(conds, enc.SegmentTokenIds[s], langId);
                codes = ar.Generate(logits, ar.PromptLen, maxSteps, rng, repPen, topP, temp, topK);
            }
            Console.WriteLine($"  [{s + 1}/{enc.Segments.Count}] AR: {codes.Length} mel codes ({t0.ElapsedMilliseconds} ms)");

            var sinfer = CodecDecode(dev, codec, codes);
            int targetLen = (int)(codes.Length * 2 * 1.72 * durationFactor);   // the codec doubles the time axis
            var cond = LengthReg(w, dev, lr, sinfer, targetLen);
            var catCond = new float[(refMelLen + targetLen) * 512];
            Array.Copy(promptCond, catCond, promptCond.Length);
            Array.Copy(cond, 0, catCond, promptCond.Length, cond.Length);

            var z = new float[(refMelLen + targetLen) * 80];
            for (int i = 0; i < z.Length; i++) z[i] = (float)(Gauss(rng));

            var vcFull = cfm.Inference(z, FlattenT(refMel), refMelLen, catCond, style, cfgRate, diffusionSteps);
            int vcLen = (refMelLen + targetLen) - refMelLen;
            var vc = new float[vcLen * 80];
            Array.Copy(vcFull, refMelLen * 80, vc, 0, vc.Length);

            var pcm = BigVganRun(dev, big, vc, vcLen);
            pieces.Add(pcm);
            Console.WriteLine($"  [{s + 1}/{enc.Segments.Count}] mel {codes.Length}->{targetLen} f0 {vcLen} -> {pcm.Length / 22050.0:F2}s ({t0.ElapsedMilliseconds} ms)");
        }

        var all = Concat(pieces);
        Program.SaveWav(outPath, all, 22050);
        Console.WriteLine($"wrote {outPath}  {all.Length / 22050.0:F2} s   total {sw.ElapsedMilliseconds} ms");
        return 0;
    }

    // ---- glue ------------------------------------------------------------------------------

    private static float[] Style(GgufWeights w, Device dev, float[,] fb, int nf)
    {
        var f = (float[,])fb.Clone();
        Dsp.SubtractColumnMean(f, nf, fb.GetLength(1));
        using var cp = new CampPlus(dev, w);
        using var g = new Graph(_rt, dev, 300000);
        g.Enter();
        var t = cp.Forward(g, g.Input(new long[] { nf, fb.GetLength(1) }, Flat(f))).MarkOutput();
        var r = t.ToFloats(g);
        g.Exit();
        return r;
    }

    /// <summary>Wav2Vec2-BERT's per-dimension mean/std for standardising hidden_states[17].
    /// Baked into the GGUF by the converter, so the CLI has no side files to depend on.</summary>
    private static (float[], float[]) LoadW2vStats(GgufWeights w)
        => (w["w2v.stats_mean"].ReadFloats(), w["w2v.stats_std"].ReadFloats());

    private static float[] SpeakerEmbedding(GgufWeights w, Device dev, float[,] feats, float[] mask,
                                            float[] mean, float[] std)
    {
        int t = feats.GetLength(0);
        var model = new W2vBert(w);
        using var g = new Graph(_rt, dev, 300000);
        g.Enter();
        var h = model.Forward(g, g.Input(new long[] { t, feats.GetLength(1) }, Flat(feats)), mask,
                              mean, std).MarkOutput();
        var r = h.ToFloats(g);
        g.Exit();
        return r;
    }

    private static float[] CodecDecode(Device dev, SemanticCodec codec, int[] codes)
    {
        using var g = new Graph(_rt, dev, 200000);
        g.Enter();
        var t = codec.Decode(g, g.InputI32(new long[] { codes.Length }, codes)).MarkOutput();
        var r = t.ToFloats(g);
        g.Exit();
        return r;
    }

    private static float[] LengthReg(GgufWeights w, Device dev, LengthRegulator lr, float[] x, int ylens)
    {
        int t = x.Length / 1024;
        using var g = new Graph(_rt, dev, 200000);
        g.Enter();
        var o = lr.Forward(g, g.Input(new long[] { t, 1024 }, x), ylens).MarkOutput();
        var r = o.ToFloats(g);
        g.Exit();
        return r;
    }

    private static float[] EmotionVec(GgufWeights w, Device dev, float[] spk)
    {
        int t = spk.Length / 1024;
        using var g = new Graph(_rt, dev, 200000);
        g.Enter();
        // No separate emotion prompt: the emotion latent is the speaker conditioning itself, so
        // merge_emovec(spk, spk) collapses to get_emovec(spk).
        var e = EmoCond.GetEmovec(g, w, g.Input(new long[] { t, 1024 }, spk), t).MarkOutput();
        var r = e.ToFloats(g);
        g.Exit();
        return r;
    }

    private static float[] CondPrefix(float[] emovec)
    {
        var c = new float[3 * 1280];
        Array.Copy(emovec, c, emovec.Length);      // rows 1-2 stay zero
        return c;
    }

    private static float[] BigVganRun(Device dev, BigVgan big, float[] vc, int vcLen)
    {
        using var g = new Graph(_rt, dev, 300000);
        g.Enter();
        var o = big.Forward(g, g.Input(new long[] { vcLen, 80 }, vc)).MarkOutput();
        var r = o.ToFloats(g);
        g.Exit();
        return r;
    }

    private static float[] LoadWav(string path, out int sampleRate)
    {
        using var r = new NAudio.Wave.AudioFileReader(path);
        sampleRate = r.WaveFormat.SampleRate;
        int ch = r.WaveFormat.Channels;
        var buf = new List<float>();
        var tmp = new float[4096];
        int n;
        while ((n = r.Read(tmp, 0, tmp.Length)) > 0)
            for (int i = 0; i < n; i++) buf.Add(tmp[i]);
        var all = buf.ToArray();
        if (ch == 1) return all;
        var mono = new float[all.Length / ch];
        for (int i = 0; i < mono.Length; i++) mono[i] = all[i * ch];
        return mono;
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

    /// <summary>mel is [80, T]; the rest of the pipeline wants [T, C].</summary>
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
        int n = parts.Sum(p => p.Length) + Math.Max(0, parts.Count - 1) * 4410;   // 0.2 s gaps
        var r = new float[n];
        int at = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            Array.Copy(parts[i], 0, r, at, parts[i].Length);
            at += parts[i].Length;
            if (i < parts.Count - 1) at += 4410;
        }
        return r;
    }
}
