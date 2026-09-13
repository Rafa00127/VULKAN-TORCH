using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using IndexTts;
using VulkanTorch;

internal static class Program
{
    internal static string FindRoot()
    {
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            if (File.Exists(Path.Combine(d.FullName, "CMakeLists.txt")) &&
                Directory.Exists(Path.Combine(d.FullName, "src")))
                return d.FullName;
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("repo root (CMakeLists.txt + src/) not found above the exe");
    }

    internal static void SaveWav(string path, float[] s, int sampleRate)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var w = new NAudio.Wave.WaveFileWriter(path, new NAudio.Wave.WaveFormat(sampleRate, 16, 1));
        var bytes = new byte[s.Length * 2];
        for (int i = 0; i < s.Length; i++)
        {
            short pcm = (short)Math.Round(Math.Clamp(s[i], -1f, 1f) * 32767f);
            bytes[2 * i] = (byte)pcm;
            bytes[2 * i + 1] = (byte)(pcm >> 8);
        }
        w.Write(bytes, 0, bytes.Length);
    }

    internal static void Report(string tag, Npy want, float[] got)
    {
        int n = (int)Math.Min(want.Numel, got.Length);
        double maxErr = 0, se = 0, sRef = 0;
        for (int i = 0; i < n; i++)
        {
            double d = want.Data[i] - got[i];
            maxErr = Math.Max(maxErr, Math.Abs(d));
            se += d * d;
            sRef += (double)want.Data[i] * want.Data[i];
        }
        double fro = Math.Sqrt(se / (sRef + 1e-12));
        float gmax = 0; for (int i = 0; i < n; i++) gmax = Math.Max(gmax, Math.Abs(got[i]));
        Console.WriteLine($"  [{tag}] n={n} max_err={maxErr:E3} fro_rel={fro:E3} got_max={gmax:E3}");
    }

    internal static void Report(string tag, float[] want, float[] got)
    {
        int n = Math.Min(want.Length, got.Length);
        double maxErr = 0, se = 0, sRef = 0;
        for (int i = 0; i < n; i++)
        {
            double d = want[i] - got[i];
            maxErr = Math.Max(maxErr, Math.Abs(d));
            se += d * d;
            sRef += (double)want[i] * want[i];
        }
        Console.WriteLine($"  [{tag}] n={n} max_err={maxErr:E3} fro_rel={Math.Sqrt(se / (sRef + 1e-12)):E3}");
    }

    private static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "synth") return Cli.Run(args);

        string mode = args.Length > 0 ? args[0] : "gpt2";
        string root = FindRoot();
        string refs = Path.Combine(root, "data", "indextts_ref");
        string gguf = Path.Combine(root, "model", "indextts2.5", "indextts2.5.f16.gguf");

        using var rt = new Runtime();
        var dev = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");

        if (mode == "gpt2") return RunGpt2(rt, dev, refs, gguf);
        if (mode == "gpt2layer") return RunGpt2Layer(rt, dev, refs, gguf);
        if (mode == "flashdiag") return RunFlashDiag(rt, dev, refs);
        if (mode == "conformer") return RunConformer(rt, dev, refs, gguf);
        if (mode == "perceiver") return RunPerceiver(rt, dev, refs, gguf);
        if (mode == "spkemo") return RunSpkEmo(rt, dev, refs, gguf);
        if (mode == "gptar") return RunGptAr(rt, dev, refs, gguf);
        if (mode == "sampler") return RunSampler(refs);
        if (mode == "gptkv") return RunGptKv(rt, dev, refs, gguf);
        if (mode == "argen") return RunArGen(rt, dev, refs, gguf);
        if (mode == "tok") return RunTok(refs);
        if (mode == "textfrontend") return RunTextFrontend(refs);
        if (mode == "codec") return RunCodec(rt, dev, refs, gguf);
        if (mode == "lenreg") return RunLenReg(rt, dev, refs, gguf);
        if (mode == "dit") return RunDit(rt, dev, refs, gguf);
        if (mode == "cfm") return RunCfm(rt, dev, refs, gguf);
        if (mode == "bigvgan") return RunBigVgan(rt, dev, refs, gguf);
        if (mode == "refaudio") return RunRefAudio(refs);
        if (mode == "campplus") return RunCampPlus(rt, dev, refs, gguf);
        if (mode == "w2v") return RunW2v(rt, dev, refs, gguf);
        if (mode == "bvup") return RunBvUp(rt, dev, refs, gguf);
        if (mode == "textnorm")
        {
            string p = args.Length > 1 ? args[1] : Path.Combine(root, "data", "_ref_textnorm", "ref.txt");
            if (!Path.IsPathRooted(p)) p = Path.Combine(root, p);
            return RunTextNorm(p);
        }
        Console.Error.WriteLine($"unknown mode '{mode}'");
        return 2;
    }

    /// <summary>Validate the C# text normalization against the dump produced by the original C++.
    /// The TSV holds four records per input: IN, EN, ZH (IndexTTS target) and PU.</summary>
    private static int RunTextNorm(string path)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        // Mirror std::getline: split on '\n' only, then strip a trailing '\r'.
        var raw = File.ReadAllText(path).Split('\n');
        int n = raw.Length;
        if (n > 0 && raw[n - 1].Length == 0) n--;   // trailing newline -> no phantom last line
        var lines = new List<string>(n);
        for (int i = 0; i < n; i++)
        {
            var l = raw[i];
            lines.Add(l.Length > 0 && l[l.Length - 1] == '\r' ? l.Substring(0, l.Length - 1) : l);
        }

        var opts = new TextNormalization.EnglishOptions
        {
            ExpandCommonContractions = true,
            SpellNumbers = true,
            IndexTtsPunctuation = true,
            UppercaseAscii = false,
            VerbalizeSymbols = true,
        };

        int ok = 0, total = 0;
        for (int g = 0; g + 3 < lines.Count; g += 4)
        {
            string input = RecordValue(lines[g]);
            string expEn = RecordValue(lines[g + 1]);
            string expZh = RecordValue(lines[g + 2]);
            string expPu = RecordValue(lines[g + 3]);

            string gotEn = TextNormalization.NormalizeEnglishText(input, opts);
            string gotZh = TextNormalization.NormalizeChineseText(input, true);
            string gotPu = TextNormalization.NormalizeIndexTtsPunctuation(input);

            bool okEn = gotEn == expEn, okZh = gotZh == expZh, okPu = gotPu == expPu;
            if (okEn) ok++;
            if (okZh) ok++;
            if (okPu) ok++;
            total += 3;

            if (okEn && okZh && okPu)
            {
                Console.WriteLine($"[PASS] {input}");
                continue;
            }
            Console.WriteLine($"[FAIL] {input}");
            if (!okEn) { Console.WriteLine($"  EN exp: {expEn}"); Console.WriteLine($"  EN got: {gotEn}"); }
            if (!okZh) { Console.WriteLine($"  ZH exp: {expZh}"); Console.WriteLine($"  ZH got: {gotZh}"); }
            if (!okPu) { Console.WriteLine($"  PU exp: {expPu}"); Console.WriteLine($"  PU got: {gotPu}"); }
        }
        Console.WriteLine($"{ok}/{total} records match");
        return ok == total ? 0 : 1;
    }

    /// <summary>Value after the first TAB of a "KEY\tvalue" record (input/output may contain tabs).</summary>
    private static string RecordValue(string line)
    {
        int t = line.IndexOf('\t');
        return t < 0 ? line : line.Substring(t + 1);
    }

    /// <summary>Validate the v2.5 text front-end (segmentation + pronunciation annotations +
    /// token ids) against dumps from the official Python. Also runs it with normalization on,
    /// which should agree on these normalizer-neutral inputs.</summary>
    private static int RunTextFrontend(string refs)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        string vocab = Path.Combine(FindRoot(), "model", "indextts2.5",
                                    "multilingual_zh_ja_yue_char_del.tiktoken");
        var tok = new IndexTokenizer(vocab);
        var front = new TextFrontend(tok);

        var cases = System.Text.Json.JsonSerializer.Deserialize<List<TfCase>>(
            File.ReadAllText(Path.Combine(refs, "textfrontend_ref.json")))!;

        int badNoTn = 0, badTn = 0;
        foreach (var c in cases)
        {
            bool okNo = CompareFrontend(front, tok, c, textNormalization: false, verbose: true, tag: "notn");
            if (!okNo) badNoTn++;
            bool okTn = CompareFrontend(front, tok, c, textNormalization: true, verbose: false, tag: "tn");
            if (!okTn) badTn++;
        }
        Console.WriteLine($"front-end (TN off): {cases.Count - badNoTn}/{cases.Count} match" +
                          $"   (TN on): {cases.Count - badTn}/{cases.Count} match");
        return badNoTn == 0 ? 0 : 1;
    }

    private sealed class TfCase
    {
        public string lang { get; set; } = "";
        public string text { get; set; } = "";
        public string processed { get; set; } = "";
        public List<string> segments { get; set; } = new();
        public List<int[]> ids { get; set; } = new();
    }

    private static bool CompareFrontend(TextFrontend front, IndexTokenizer tok, TfCase c,
                                        bool textNormalization, bool verbose, string tag)
    {
        var got = front.EncodeForInference(c.text, TextFrontend.DefaultMaxTokensPerSegment,
                                           c.lang, textNormalization);
        bool ok = got.NormalizedText == c.processed
                  && got.Segments.SequenceEqual(c.segments)
                  && got.SegmentTokenIds.Count == c.ids.Count
                  && got.SegmentTokenIds.Zip(c.ids).All(p => p.First.SequenceEqual(p.Second));
        if (!ok && verbose)
        {
            Console.WriteLine($"[FAIL:{tag}] [{c.lang}] {Trunc(c.text)}");
            if (got.NormalizedText != c.processed)
            {
                Console.WriteLine($"  text exp: {Trunc(c.processed)}");
                Console.WriteLine($"  text got: {Trunc(got.NormalizedText)}");
            }
            Console.WriteLine($"  segs exp: {string.Join(" | ", c.segments.Select(Trunc))}");
            Console.WriteLine($"  segs got: {string.Join(" | ", got.Segments.Select(Trunc))}");
            Console.WriteLine($"  ids  exp: {string.Join(" ; ", c.ids.Select(a => string.Join(",", a.Take(8))))}");
            Console.WriteLine($"  ids  got: {string.Join(" ; ", got.SegmentTokenIds.Select(a => string.Join(",", a.Take(8))))}");
        }
        return ok;
    }

    private static string Trunc(string s) => s.Length <= 48 ? s : s.Substring(0, 48) + "…";

    /// <summary>Validate the semantic codec decode chain against the reference dump
    /// (codec_quant / codec_dec / codec_xr .npy).</summary>
    private static int RunCodec(Runtime rt, Device dev, string refs, string gguf)
    {
        var codes = Npy.Load(Path.Combine(refs, "codec_codes.npy")).Data.Select(f => (int)f).ToArray();
        var wantQuant = Transpose1CT(Npy.Load(Path.Combine(refs, "codec_quant.npy")));
        var wantDec = Npy.Load(Path.Combine(refs, "codec_dec.npy"));
        var wantXr = Npy.Load(Path.Combine(refs, "codec_xr.npy"));
        Console.WriteLine($"codes={codes.Length} quant={string.Join("x", wantQuant.Shape)} " +
                          $"dec={string.Join("x", wantDec.Shape)} xr={string.Join("x", wantXr.Shape)}");

        using var w = new GgufWeights(gguf, dev, new[] { "codec." });
        Console.Error.WriteLine($"loaded {w.Count} codec tensors");
        using var codec = new SemanticCodec(dev, w);
        using var g = new Graph(rt, dev);
        g.Enter();
        var codesT = g.InputI32(new long[] { codes.Length }, codes);
        var (q, finln, blocks, dec, rec) = codec.DecodeStages(g, codesT);
        q.MarkOutput(); finln.MarkOutput(); dec.MarkOutput(); rec.MarkOutput();
        foreach (var b in blocks) b.MarkOutput();
        Report("quant", wantQuant, q.ToFloats(g));
        for (int i = 0; i < blocks.Count; i++)
        {
            // block hook output is torch [B, C, T]
            var want = Transpose1CT(Npy.Load(Path.Combine(refs, $"codec_blk{i}.npy")));
            Report($"blk{i}", want, blocks[i].ToFloats(g));
        }
        Report("finln", Npy.Load(Path.Combine(refs, "codec_finln.npy")), finln.ToFloats(g));
        Report("dec", wantDec, dec.ToFloats(g));
        Report("xr", wantXr, rec.ToFloats(g));
        g.Exit();
        return 0;
    }

    /// <summary>Validate the s2mel length_regulator against the reference dump
    /// (lr_proj / lr_up / lr_out .npy).</summary>
    private static int RunLenReg(Runtime rt, Device dev, string refs, string gguf)
    {
        var sInfer = Npy.Load(Path.Combine(refs, "lr_sinfer.npy"));     // [1, T, 1024]
        int ylens = (int)Npy.Load(Path.Combine(refs, "lr_ylens.npy")).Data[0];
        var wantProj = Npy.Load(Path.Combine(refs, "lr_proj.npy"));
        var wantUp = Npy.Load(Path.Combine(refs, "lr_up.npy"));
        var wantOut = Npy.Load(Path.Combine(refs, "lr_out.npy"));
        int t = (int)sInfer.Shape[1];
        Console.WriteLine($"ylens={ylens} T={t} out={string.Join("x", wantOut.Shape)}");

        using var w = new GgufWeights(gguf, dev, new[] { "s2mel.length_regulator." });
        var lr = new LengthRegulator(w);
        using var g = new Graph(rt, dev);
        g.Enter();
        var x = g.Input(new long[] { t, 1024 }, sInfer.Data);
        var proj = lr.Project(g, x);
        var up = lr.Nearest(g, proj, t, ylens);
        var outT = lr.Forward(g, x, ylens);
        proj.MarkOutput(); up.MarkOutput(); outT.MarkOutput();
        Report("proj", wantProj, proj.ToFloats(g));
        Report("up", wantUp, up.ToFloats(g));
        Report("out", wantOut, outT.ToFloats(g));
        g.Exit();
        return 0;
    }

    /// <summary>Validate the s2mel DiT estimator against the reference dump (dit_*.npy).</summary>
    private static int RunDit(Runtime rt, Device dev, string refs, string gguf)
    {
        var x = Transpose1CT(Npy.Load(Path.Combine(refs, "dit_x.npy")));            // [T, 80]
        var promptX = Transpose1CT(Npy.Load(Path.Combine(refs, "dit_prompt_x.npy")));
        var cond = Npy.Load(Path.Combine(refs, "dit_cond.npy"));                    // [1, T, 512]
        var style = Npy.Load(Path.Combine(refs, "dit_style.npy"));                  // [1, 192]
        var wantOut = Transpose1CT(Npy.Load(Path.Combine(refs, "dit_out.npy")));    // [T, 80]
        float t = Npy.Load(Path.Combine(refs, "dit_t.npy")).Data[0];
        int n = (int)x.Shape[0];
        Console.WriteLine($"T={n} t={t} out={string.Join("x", wantOut.Shape)}");

        using var w = new GgufWeights(gguf, dev, new[] { "s2mel.cfm.estimator." });
        using var dit = new Dit(dev, w);
        using var g = new Graph(rt, dev);
        g.Enter();
        var pos = g.InputI32(new long[] { n }, Enumerable.Range(0, n).ToArray());
        var xT = g.Input(x.Shape, x.Data);
        var pT = g.Input(promptX.Shape, promptX.Data);
        var cT = g.Input(cond.Shape, cond.Data);
        var sT = g.Input(style.Shape, style.Data);
        var trace = new List<Tensor>();
        var outT = dit.Forward(g, xT, pT, cT, sT, t, pos, trace);
        outT.MarkOutput();
        foreach (var tb in trace) tb.MarkOutput();
        foreach (int i in new[] { 0, 6, 12 })
            Report($"blk{i}", Npy.Load(Path.Combine(refs, $"dit_blk{i}.npy")), trace[i].ToFloats(g));
        Report("out", wantOut, outT.ToFloats(g));
        g.Exit();
        return 0;
    }

    /// <summary>Validate the s2mel CFM (25-step Euler + CFG) against cfm_*.npy.</summary>
    private static int RunCfm(Runtime rt, Device dev, string refs, string gguf)
    {
        var z = Transpose1CT(Npy.Load(Path.Combine(refs, "cfm_z.npy")));            // [T, 80]
        var prompt = Transpose1CT(Npy.Load(Path.Combine(refs, "cfm_prompt.npy")));  // [Tp, 80]
        var mu = Npy.Load(Path.Combine(refs, "cfm_mu.npy"));                        // [1, T, 512]
        var style = Npy.Load(Path.Combine(refs, "cfm_style.npy"));                  // [1, 192]
        var want = Transpose1CT(Npy.Load(Path.Combine(refs, "cfm_out.npy")));       // [T, 80]
        int n = (int)z.Shape[0], tp = (int)prompt.Shape[0];
        Console.WriteLine($"T={n} promptLen={tp} out={string.Join("x", want.Shape)}");

        using var w = new GgufWeights(gguf, dev, new[] { "s2mel.cfm.estimator." });
        using var dit = new Dit(dev, w);
        var cfm = new Cfm(rt, dev, dit);
        var got = cfm.Inference(z.Data, prompt.Data, tp, mu.Data, style.Data, 0.7f, 25);
        Report("cfm", want, got);
        return 0;
    }

    /// <summary>CAMPPlus style vector from the reference audio's Kaldi fbank.</summary>
    private static int RunCampPlus(Runtime rt, Device dev, string refs, string gguf)
    {
        var w16 = Npy.Load(Path.Combine(refs, "ra_wav16k.npy"));
        var wav = new double[w16.Numel];
        for (int i = 0; i < wav.Length; i++) wav[i] = w16.Data[i];
        var fb = Dsp.KaldiFbank(wav, 16000, 80);
        int t = fb.GetLength(0), bins = fb.GetLength(1);
        Dsp.SubtractColumnMean(fb, t, bins);
        Console.WriteLine($"fbank {t}x{bins}");

        using var w = new GgufWeights(gguf, dev, new[] { "campplus." });
        using var cp = new CampPlus(dev, w);
        using var g = new Graph(rt, dev, 200000);
        g.Enter();
        var dbg = new Dictionary<string, Tensor>();
        var style = cp.Forward(g, g.Input(new long[] { t, bins }, Flat(fb)), dbg).MarkOutput();
        foreach (var v in dbg.Values) v.MarkOutput();
        foreach (var k in new[] { "head", "tdnn", "blk1", "tr1", "blk2", "tr2", "blk3", "outnl", "dense" })
            Report(k, Transpose1CT(Npy.Load(Path.Combine(refs, $"ra_cp_{k}.npy"))), dbg[k].ToFloats(g));
        Report("style", Npy.Load(Path.Combine(refs, "ra_style.npy")), style.ToFloats(g));
        Console.WriteLine($"  nodes={g.NodeCount}");
        g.Exit();
        return 0;
    }

    /// <summary>Wav2Vec2Bert hidden_states[17] and the standardised speaker embedding.</summary>
    private static int RunW2v(Runtime rt, Device dev, string refs, string gguf)
    {
        var feats = Npy.Load(Path.Combine(refs, "ra_feats.npy"));              // [1, T, 160]
        int t = (int)feats.Shape[1];
        var mask = Npy.Load(Path.Combine(refs, "ra_mask.npy")).Data;
        var mean = Npy.Load(Path.Combine(refs, "ra_w2v_mean.npy")).Data;
        var std = Npy.Load(Path.Combine(refs, "ra_w2v_std.npy")).Data;
        Console.WriteLine($"feats {t}x160");

        using var w = new GgufWeights(gguf, dev, new[] { "w2v." });
        var model = new W2vBert(w);
        using var g = new Graph(rt, dev, 200000);
        g.Enter();
        var dbg = new Dictionary<string, Tensor>();
        var h = model.Forward(g, g.Input(new long[] { t, 160 }, feats.Data), mask, mean, std, dbg).MarkOutput();
        foreach (var v in dbg.Values) v.MarkOutput();
        foreach (var k in new[] { "hs0", "hs1", "hs2", "hs5", "hs9", "hs17" })
            Report(k, Npy.Load(Path.Combine(refs, $"ra_{k}.npy")), dbg[k].ToFloats(g));
        Report("spk", Npy.Load(Path.Combine(refs, "ra_spk.npy")), h.ToFloats(g));
        Console.WriteLine($"  nodes={g.NodeCount}");
        g.Exit();
        return 0;
    }

    private static float[] Flat(float[,] a)
    {
        var r = new float[a.Length];
        Buffer.BlockCopy(a, 0, r, 0, a.Length * 4);
        return r;
    }

    /// <summary>Host DSP front-end: Kaldi fbank and the SeamlessM4T features.</summary>
    private static int RunRefAudio(string refs)
    {
        var w16 = Npy.Load(Path.Combine(refs, "ra_wav16k.npy"));
        var wav = new double[w16.Numel];
        for (int i = 0; i < wav.Length; i++) wav[i] = w16.Data[i];
        Console.WriteLine($"wav16k n={wav.Length}");

        var fb = Dsp.KaldiFbank(wav, 16000, 80);
        int frames = fb.GetLength(0), bins = fb.GetLength(1);
        Console.WriteLine($"fbank {frames}x{bins}");
        Dsp.SubtractColumnMean(fb, frames, bins);
        Report("fbank", Npy.Load(Path.Combine(refs, "ra_fbank.npy")), Flat(fb));

        var feats = Dsp.SeamlessFeatures(wav);
        Console.WriteLine($"feats {feats.GetLength(0)}x{feats.GetLength(1)}");
        Report("feats", Npy.Load(Path.Combine(refs, "ra_feats.npy")), Flat(feats));

        var w22 = Npy.Load(Path.Combine(refs, "ra_wav22k.npy"));
        var wav22 = new double[w22.Numel];
        for (int i = 0; i < wav22.Length; i++) wav22[i] = w22.Data[i];
        var mel = Dsp.MelSpectrogram(wav22);
        Console.WriteLine($"mel {mel.GetLength(0)}x{mel.GetLength(1)}");
        Report("mel", Npy.Load(Path.Combine(refs, "ra_mel.npy")), Flat(mel));
        return 0;
    }

    /// <summary>Isolated upsample stage: bv_pre -> ups.0.0 -> bv_up0.</summary>
    private static int RunBvUp(Runtime rt, Device dev, string refs, string gguf)
    {
        var pre = Transpose1CT(Npy.Load(Path.Combine(refs, "bv_pre.npy")));
        var want = Transpose1CT(Npy.Load(Path.Combine(refs, "bv_up0.npy")));
        using var w = new GgufWeights(gguf, dev, new[] { "bigvgan." });
        using var bv = new BigVgan(dev, w);
        using var g = new Graph(rt, dev);
        g.Enter();
        var o = bv.Up(g, g.Input(pre.Shape, pre.Data), "ups.0.0", 4).MarkOutput();
        Report("up0", want, o.ToFloats(g));
        g.Exit();
        return 0;
    }

    /// <summary>Validate the BigVGAN generator (mel -> wav) against bv_*.npy.</summary>
    private static int RunBigVgan(Runtime rt, Device dev, string refs, string gguf)
    {
        var x = Transpose1CT(Npy.Load(Path.Combine(refs, "bv_x.npy")));             // [T, 80]
        var want = Npy.Load(Path.Combine(refs, "bv_out.npy"));                      // [1,1,L]
        int n = (int)x.Shape[0];
        Console.WriteLine($"T={n} out_numel={want.Numel}");

        using var w = new GgufWeights(gguf, dev, new[] { "bigvgan." });
        using var bv = new BigVgan(dev, w);
        using var g = new Graph(rt, dev);
        g.Enter();
        var trace = new List<Tensor>();
        var dbg = new Dictionary<string, Tensor>();
        var outT = bv.Forward(g, g.Input(x.Shape, x.Data), trace, dbg).MarkOutput();
        foreach (var t in trace) t.MarkOutput();
        foreach (var t in dbg.Values) t.MarkOutput();
        if (Environment.GetEnvironmentVariable("BV_STATIC") != null) { g.AllocStatic(); g.ComputeStatic(); }
        foreach (var kv in new[]
        {
            ("a0up", "resblocks.0.activations.0.up"), ("a0act", "resblocks.0.activations.0.act"),
            ("a0", "resblocks.0.activations.0.down"), ("c10", "resblocks.0.c1.0"),
            ("c20", "resblocks.0.c2.0"), ("a2", "resblocks.0.activations.2.down"),
            ("c11", "resblocks.0.c1.1"), ("a3", "resblocks.0.activations.3.down"),
            ("c21", "resblocks.0.c2.1"), ("c22", "resblocks.0.c2.2"),
        })
        {
            var pth = Path.Combine(refs, $"bv_{kv.Item1}.npy");
            if (File.Exists(pth))
                Report(kv.Item1, Transpose1CT(Npy.Load(pth)), dbg[kv.Item2].ToFloats(g));
        }
        for (int k = 0; k < 6; k++)
            Report($"stage{k}", Transpose1CT(Npy.Load(Path.Combine(refs, $"bv_stage{k}.npy"))), dbg[$"stage{k}"].ToFloats(g));
        Report("pre", Transpose1CT(Npy.Load(Path.Combine(refs, "bv_pre.npy"))), trace[0].ToFloats(g));
        Report("up0", Transpose1CT(Npy.Load(Path.Combine(refs, "bv_up0.npy"))), trace[1].ToFloats(g));
        Report("rb0", Transpose1CT(Npy.Load(Path.Combine(refs, "bv_rb0.npy"))), trace[2].ToFloats(g));
        Console.WriteLine("  trace[2]==dbg[x.2]? " + ReferenceEquals(trace[2], dbg["resblocks.0.x.2"]));
        Report("x.2(dbg)", Transpose1CT(Npy.Load(Path.Combine(refs, "bv_rb0.npy"))), dbg["resblocks.0.x.2"].ToFloats(g));
        Report("x.0(dbg)", Transpose1CT(Npy.Load(Path.Combine(refs, "bv_rb0.npy"))), dbg["resblocks.0.x.0"].ToFloats(g));
        Report("actpost", Transpose1CT(Npy.Load(Path.Combine(refs, "bv_actpost.npy"))), trace[3].ToFloats(g));
        Report("wav", want, outT.ToFloats(g));
        g.Exit();
        return 0;
    }

    /// <summary>Reference dumps use torch [1, C, T]; our tensors are PT [T, C]. Reorder.</summary>
    private static Npy Transpose1CT(Npy a)
    {
        if (a.Shape.Length != 3 || a.Shape[0] != 1) return a;
        int c = (int)a.Shape[1], t = (int)a.Shape[2];
        var d = new float[c * t];
        for (int ci = 0; ci < c; ci++)
            for (int ti = 0; ti < t; ti++)
                d[ti * c + ci] = a.Data[ci * t + ti];
        return Npy.Wrap(new long[] { t, c }, d);
    }

    /// <summary>Validate the text tokenizer against ids dumped from the reference tiktoken.</summary>
    private static int RunTok(string refs)
    {
        string vocab = Path.Combine(FindRoot(), "model", "indextts2.5",
                                    "multilingual_zh_ja_yue_char_del.tiktoken");
        var tok = new IndexTokenizer(vocab);
        var want = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, int[]>>(
            File.ReadAllText(Path.Combine(refs, "tok_ref.json")))!;
        int bad = 0;
        foreach (var (text, ids) in want)
        {
            var got = tok.Encode(text);
            bool ok = got.SequenceEqual(ids);
            Console.WriteLine($"  [{(ok ? "OK " : "FAIL")}] n_ref={ids.Length,3} n_got={got.Length,3}  {text}");
            if (!ok)
            {
                bad++;
                Console.WriteLine($"        ref={string.Join(",", ids.Take(10))}");
                Console.WriteLine($"        got={string.Join(",", got.Take(10))}");
            }
        }
        Console.WriteLine(bad == 0 ? "  tokenizer matches the reference" : $"  {bad} MISMATCH");
        return 0;
    }

    /// <summary>Validate the AR decoder (prefill + replayed step) and generate mel codes.</summary>
    private static int RunArGen(Runtime rt, Device dev, string refs, string gguf)
    {
        var conds = Npy.Load(Path.Combine(refs, "ar_conds.npy"));
        var tids = Npy.Load(Path.Combine(refs, "ar_text_inputs.npy")).Data.Select(f => (int)f).ToArray();
        int nextTok = (int)Npy.Load(Path.Combine(refs, "ar_next_token.npy")).Data[0];
        var wantPre = Npy.Load(Path.Combine(refs, "ar_logits_prefill.npy"));   // [18,8194]

        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var dec = new ArDecoder(rt, dev, w, 4096);
        var lg = dec.Prefill(conds.Data, tids, 0);
        Console.WriteLine($"melLen={dec.MelLen} promptLen={dec.PromptLen}");

        var wantLast = new float[Gpt2.MelVocab];
        Array.Copy(wantPre.Data, (dec.PromptLen - 1) * Gpt2.MelVocab, wantLast, 0, Gpt2.MelVocab);
        Report("ar_prefill_last", wantLast, lg);

        var s1 = dec.Step(nextTok, dec.PromptLen);
        Report("ar_step1", Npy.Load(Path.Combine(refs, "ar_logits_step1.npy")), s1);

        // the captured step graph is replayed for every token, so a replay must be bit-exact
        var s2 = dec.Step(nextTok, dec.PromptLen);
        double d = 0;
        for (int i = 0; i < s1.Length; i++) d = Math.Max(d, Math.Abs(s1[i] - s2[i]));
        Console.WriteLine($"  replay bit-exact: maxdiff={d:E3}  (must be 0)");

        int maxNew = int.TryParse(Environment.GetEnvironmentVariable("MAXNEW"), out var mn) ? mn : 200;
        var rng = new Random(42);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var codes = dec.Generate(s1, dec.PromptLen + 1, maxNew, rng);
        sw.Stop();
        Console.WriteLine($"  generated {codes.Length} codes in {sw.ElapsedMilliseconds} ms "
                          + $"({(codes.Length > 0 ? sw.Elapsed.TotalMilliseconds / codes.Length : 0):F1} ms/code)");
        Console.WriteLine($"  first 16: {string.Join(",", codes.Take(16))}");
        Console.WriteLine($"  in-range: {codes.Count(c => c >= 0 && c < Gpt2.MelVocab)}/{codes.Length}");
        return 0;
    }

    /// <summary>Validate the KV-cached prefill + decode against the uncached reference.</summary>
    private static int RunGptKv(Runtime rt, Device dev, string refs, string gguf)
    {
        var conds = Npy.Load(Path.Combine(refs, "ar_conds.npy"));
        var tidsN = Npy.Load(Path.Combine(refs, "ar_text_inputs.npy"));
        int[] tids = tidsN.Data.Select(f => (int)f).ToArray();
        int nextTok = (int)Npy.Load(Path.Combine(refs, "ar_next_token.npy")).Data[0];

        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var kv = Gpt2Cached.NewCache(dev, 4096);
        float[] l1, l2;
        int t, melLen;

        // ---- prefill: writes rows 0..t-1 ----
        using (var g = new Graph(rt, dev))
        {
            g.Enter();
            var condsT = g.Input(new long[] { GptAr.CondRows, Gpt2.D }, conds.Data);
            var inputsEmbeds = GptAr.PrepareGptInputs(g, w, condsT, tids, 0, out melLen);
            var prefill = Ops.Concat(inputsEmbeds, GptAr.MelToken(g, w, GptAr.StartMel, 0), 0);
            t = (int)prefill.Shape[0];
            int lk = GraphCache.PickBucket(t);
            var pos = g.InputI32(new long[] { t }, Enumerable.Range(0, t).ToArray());
            var mask = Ops.Cast(g.Input(new long[] { t, lk }, GraphCache.CausalMask(t, lk, 0)), Ops.F16);
            var fn = Gpt2Cached.ForwardCached(w, prefill, kv, pos, mask, lk, t);
            var lg = Ops.Contiguous(Gpt2.MelHead(w, fn)).MarkOutput();
            l1 = lg.ToFloats(g);
            g.Exit();
            Console.WriteLine($"prefill: t={t} lk={lk}");
        }

        // ---- one decode step: writes row t, reads window of lk ----
        using (var g = new Graph(rt, dev))
        {
            g.Enter();
            int lk = GraphCache.PickBucket(t + 1);
            var emb = GptAr.MelToken(g, w, nextTok, (t + 1) - melLen);   // PT [1, D]
            var pos = g.InputI32(new long[] { 1 }, new[] { t });
            var mask = Ops.Cast(g.Input(new long[] { 1, lk }, GraphCache.CausalMask(1, lk, t)), Ops.F16);
            var fn = Gpt2Cached.ForwardCached(w, emb, kv, pos, mask, lk, 1);
            var lg = Ops.Contiguous(Gpt2.MelHead(w, fn)).MarkOutput();
            l2 = lg.ToFloats(g);
            g.Exit();
            Console.WriteLine($"decode : t=1 lk={lk} pos={t}");
        }

        var wantPre = Npy.Load(Path.Combine(refs, "ar_logits_prefill.npy"));
        Report("kv_logits_prefill", wantPre, l1);
        int agree = 0;
        for (int i = 0; i < t; i++)
            if (ArgMax(wantPre.Data, i * Gpt2.MelVocab, Gpt2.MelVocab) ==
                ArgMax(l1, i * Gpt2.MelVocab, Gpt2.MelVocab)) agree++;
        Console.WriteLine($"  prefill argmax agreement: {agree}/{t}");
        Report("kv_logits_step1", Npy.Load(Path.Combine(refs, "ar_logits_step1.npy")), l2);
        return 0;
    }

    /// <summary>Dump sampled probabilities for the fixed step-1 logits (compared against HF).</summary>
    private static int RunSampler(string refs)
    {
        var logits = Npy.Load(Path.Combine(refs, "ar_logits_step1.npy"));   // [8194]
        int[] codes = { 100, 200, 300, 6491 };
        var s = (float[])logits.Data.Clone();
        var smp = new Sampler { RepPenalty = 10f, TopP = 0.8f, Temperature = 0.8f, TopK = 30 };
        smp.Reset(GptAr.Penalized(codes));
        smp.LogProbs(s);
        var p = Sampler.Probabilities(s);

        var bb = new byte[p.Length * 8];
        Buffer.BlockCopy(p, 0, bb, 0, bb.Length);
        File.WriteAllBytes(Path.Combine(refs, "_cs_sampler_probs.f64"), bb);

        int finite = 0;
        foreach (var v in p) if (v > 0) finite++;
        Console.WriteLine($"  finite entries after top_k/top_p: {finite}");
        var top = Enumerable.Range(0, p.Length).OrderByDescending(i => p[i]).Take(5).ToArray();
        foreach (var i in top) Console.WriteLine($"    tok {i,5}  p={p[i]:F6}");
        return 0;
    }

    /// <summary>Validate the GPT AR input assembly + prefill logits + one decode step.</summary>
    private static int RunGptAr(Runtime rt, Device dev, string refs, string gguf)
    {
        var conds = Npy.Load(Path.Combine(refs, "ar_conds.npy"));                 // [1,3,1280]
        var tidsN = Npy.Load(Path.Combine(refs, "ar_text_inputs.npy"));           // [1,12]
        var nextN = Npy.Load(Path.Combine(refs, "ar_next_token.npy"));            // [1,1]
        int[] tids = tidsN.Data.Select(f => (int)f).ToArray();
        int nextTok = (int)nextN.Data[0];
        Console.WriteLine($"text ids: {tids.Length}, next mel token: {nextTok}");

        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var g = new Graph(rt, dev);
        g.Enter();
        var condsT = g.Input(new long[] { GptAr.CondRows, Gpt2.D }, conds.Data);

        // prefill: inputs_embeds + the start_mel embedding at mel position 0
        var inputsEmbeds = GptAr.PrepareGptInputs(g, w, condsT, tids, 0, out int melLen);
        var prefill = Ops.Concat(inputsEmbeds, GptAr.MelToken(g, w, GptAr.StartMel, 0), 0);
        int t = (int)prefill.Shape[0];

        // decode step 1: mel pos index is (attn_len - mel_len) = (t + 1) - melLen
        var step = Ops.Concat(prefill, GptAr.MelToken(g, w, nextTok, (t + 1) - melLen), 0);
        int t2 = (int)step.Shape[0];
        Console.WriteLine($"melLen={melLen} prefill={t} step={t2} (mel pos idx={(t + 1) - melLen})");

        var logits1 = Ops.Contiguous(GptAr.Logits(w, g, prefill, t)).MarkOutput();
        var logits2 = Ops.Contiguous(GptAr.Logits(w, g, step, t2)).MarkOutput();
        var emb1 = Ops.Contiguous(inputsEmbeds).MarkOutput();
        var e1 = emb1.ToFloats(g);
        var l1 = logits1.ToFloats(g);
        var l2 = logits2.ToFloats(g);
        g.Exit();

        var wantPre = Npy.Load(Path.Combine(refs, "ar_logits_prefill.npy"));
        Report("ar_inputs_embeds", Npy.Load(Path.Combine(refs, "ar_inputs_embeds.npy")), e1);
        Report("ar_logits_prefill", wantPre, l1);
        int agree = 0;
        for (int i = 0; i < t; i++)
            if (ArgMax(wantPre.Data, i * Gpt2.MelVocab, Gpt2.MelVocab) ==
                ArgMax(l1, i * Gpt2.MelVocab, Gpt2.MelVocab)) agree++;
        Console.WriteLine($"  prefill argmax agreement: {agree}/{t}");
        // the reference step-1 logits are only the *last* position
        var last = new float[Gpt2.MelVocab];
        Array.Copy(l2, (t2 - 1) * Gpt2.MelVocab, last, 0, Gpt2.MelVocab);
        var wantStep = Npy.Load(Path.Combine(refs, "ar_logits_step1.npy"));
        Report("ar_logits_step1(last)", wantStep, last);
        Console.WriteLine("  (step1 top-2 in the ref differ by only 0.006, so its argmax is a coin flip under f16 noise)");
        return 0;
    }

    /// <summary>Validate spk_emb_proj and get_emovec against PyTorch.</summary>
    private static int RunSpkEmo(Runtime rt, Device dev, string refs, string gguf)
    {
        var spkIn = Npy.Load(Path.Combine(refs, "emo_spk_in.npy"));        // [1,192]
        var emoIn = Npy.Load(Path.Combine(refs, "gpt_emo_conf_in.npy"));   // [1,40,1024]
        int te = (int)emoIn.Shape[1];
        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var g = new Graph(rt, dev);
        g.Enter();
        var spk = EmoCond.SpkEmbProj(w, g.Input(new long[] { 1, spkIn.Numel }, spkIn.Data));
        var emo = EmoCond.GetEmovec(g, w, g.Input(new long[] { te, 1024 }, emoIn.Data), te);
        var spkO = Ops.Contiguous(spk).MarkOutput();
        var emoO = Ops.Contiguous(emo).MarkOutput();
        var sv = spkO.ToFloats(g);
        var ev = emoO.ToFloats(g);
        g.Exit();
        Report("emo_spk", Npy.Load(Path.Combine(refs, "emo_spk.npy")), sv);
        Report("emo_emovec", Npy.Load(Path.Combine(refs, "emo_emovec.npy")), ev);
        return 0;
    }

    /// <summary>Validate the PerceiverResampler against PyTorch.</summary>
    private static int RunPerceiver(Runtime rt, Device dev, string refs, string gguf)
    {
        var inp = Npy.Load(Path.Combine(refs, "gpt_emo_conf_out.npy"));   // [1,19,512]
        int t = (int)inp.Shape[1];
        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var g = new Graph(rt, dev);
        g.Enter();
        var x = g.Input(new long[] { t, Perceiver.DimCtx }, inp.Data);
        var y = Ops.Contiguous(Perceiver.Forward(w, x, t)).MarkOutput();
        var yv = y.ToFloats(g);
        g.Exit();
        Report("emo_perceiver", Npy.Load(Path.Combine(refs, "gpt_emo_perceiver_out.npy")), yv);
        return 0;
    }

    /// <summary>Validate the 4-layer Conformer emotion encoder stage by stage.</summary>
    private static int RunConformer(Runtime rt, Device dev, string refs, string gguf)
    {
        var inp = Npy.Load(Path.Combine(refs, "gpt_emo_conf_in.npy"));   // [1,40,1024]
        int te = (int)inp.Shape[1];
        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var g = new Graph(rt, dev);
        g.Enter();
        var x = g.Input(new long[] { te, Conformer.In }, inp.Data);
        var (y0, pos) = Conformer.Embed(g, w, x, te);
        int t = (int)y0.Shape[0];

        // recompute the embed's conv output for a stage-level dump
        string ep = "gpt.emo_conditioning_encoder.";
        var b4 = Ops.Reshape(x, new long[] { 1, 1, te, Conformer.In });
        var emoConv = Ops.Relu(Ops.Add(Ops.Conv2d(w[ep + "embed.conv.0.weight"], b4, 2, 2, 0, 0, 1, 1),
            Ops.Reshape(w[ep + "embed.conv.0.bias"], new long[] { 1, 512, 1, 1 })));

        string p = "gpt.emo_conditioning_encoder.encoders.0.";
        var attn = Conformer.RelPosAttn(w, 0, Conformer.LnAffine(w, p + "norm_mha", y0), pos, t);
        var m1 = Ops.Add(y0, attn);
        var conv = Conformer.ConvModule(w, 0, Conformer.LnAffine(w, p + "norm_conv", m1), t);
        var m2 = Ops.Add(m1, conv);
        var ff = Conformer.Linear(w, p + "feed_forward.w_2", Ops.Silu(
            Conformer.Linear(w, p + "feed_forward.w_1", Conformer.LnAffine(w, p + "norm_ff", m2))));
        var l0 = Conformer.LnAffine(w, p + "norm_final", Ops.Add(m2, ff));

        var y1 = Conformer.Layer(w, 1, l0, pos, t);
        var y2 = Conformer.Layer(w, 2, y1, pos, t);
        var y3 = Conformer.Layer(w, 3, y2, pos, t);
        var conf = Conformer.LnAffine(w, "gpt.emo_conditioning_encoder.after_norm", y3);

        var stages = new (string Name, Tensor T)[] {
            ("emo_conv", emoConv),
            ("emo_embed", y0), ("emo_pos", pos), ("emo_l0_attn", attn), ("emo_l0_conv", conv),
            ("emo_l0_ff", ff), ("emo_l0", l0), ("emo_l1", y1), ("emo_l2", y2), ("emo_l3", y3),
            ("emo_conf", conf) };
        foreach (var (_, tt) in stages) tt.MarkOutput();
        foreach (var (name, tt) in stages)
        {
            var vv = tt.ToFloats(g);
            Report(name, Npy.Load(Path.Combine(refs, name + ".npy")), vv);
            if (name == "emo_embed" || name == "emo_pos")
            {
                var bb = new byte[vv.Length * 4];
                Buffer.BlockCopy(vv, 0, bb, 0, bb.Length);
                File.WriteAllBytes(Path.Combine(refs, $"_cs_{name}.f32"), bb);
            }
        }
        g.Exit();
        return 0;
    }

    /// <summary>
    /// Isolate the attention bug. Variants:
    ///   A ref q/k/v (already [nh,T,hd]) + causal mask -> FlashAttn
    ///   B same but no mask
    ///   C reference qkv [T,3D] -> View2d split -> ToHeads -> FlashAttn  (tests the split path)
    /// </summary>
    private static int RunFlashDiag(Runtime rt, Device dev, string refs)
    {
        var qr = Npy.Load(Path.Combine(refs, "gpt_l0_q.npy"));   // PT [nh, T, hd]
        var kr = Npy.Load(Path.Combine(refs, "gpt_l0_k.npy"));
        var vr = Npy.Load(Path.Combine(refs, "gpt_l0_v.npy"));
        int t = (int)qr.Shape[1], nh = (int)qr.Shape[0], hd = (int)qr.Shape[2];
        int d = nh * hd;
        Console.WriteLine($"q/k/v PT [{nh},{t},{hd}]");
        var wantMasked = Npy.Load(Path.Combine(refs, "gpt_l0_attn_raw.npy"));
        var wantNoMask = Npy.Load(Path.Combine(refs, "gpt_l0_attn_nomask.npy"));

        // A / B: feed reference q/k/v heads directly
        using (var g = new Graph(rt, dev))
        {
            g.Enter();
            var mask = Ops.Cast(g.Input(new long[] { t, t }, Gpt2.CausalMask(t)), Ops.F16);
            var q = g.Input(new long[] { nh, t, hd }, qr.Data);
            var k = g.Input(new long[] { nh, t, hd }, kr.Data);
            var v = g.Input(new long[] { nh, t, hd }, vr.Data);
            var am = Ops.Reshape(Ops.Contiguous(Ops.FlashAttn(q, k, v, mask, 1f / MathF.Sqrt(hd))),
                                 new long[] { t, d }).MarkOutput();
            var an = Ops.Reshape(Ops.Contiguous(Ops.FlashAttn(q, k, v, null, 1f / MathF.Sqrt(hd))),
                                 new long[] { t, d }).MarkOutput();
            var vm = am.ToFloats(g);
            var vn = an.ToFloats(g);
            g.Exit();
            Report("A_refqkv_mask", wantMasked, vm);
            Report("B_refqkv_nomask", wantNoMask, vn);
        }

        // C: reference qkv -> the exact View2d+ToHeads split the port uses
        var qkvr = Npy.Load(Path.Combine(refs, "gpt_l0_qkv.npy"));  // [1,T,3D]
        using (var g = new Graph(rt, dev))
        {
            g.Enter();
            var mask = Ops.Cast(g.Input(new long[] { t, t }, Gpt2.CausalMask(t)), Ops.F16);
            var qkv = g.Input(new long[] { t, 3 * d }, qkvr.Data);
            ulong rowBytes = (ulong)(3 * d * 4);
            var q = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, d, t, rowBytes, 0)), t, nh);
            var k = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, d, t, rowBytes, (ulong)(d * 4))), t, nh);
            var v = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, d, t, rowBytes, (ulong)(2 * d * 4))), t, nh);
            var ac = Ops.Reshape(Ops.Contiguous(Ops.FlashAttn(q, k, v, mask, 1f / MathF.Sqrt(hd))),
                                 new long[] { t, d }).MarkOutput();
            var vc = ac.ToFloats(g);
            g.Exit();
            Report("C_refqkv_split_mask", wantMasked, vc);
        }
        return 0;
    }

    /// <summary>Walk GPT2 block 0 stage by stage to isolate where the port diverges.</summary>
    private static int RunGpt2Layer(Runtime rt, Device dev, string refs, string gguf)
    {
        var emb = Npy.Load(Path.Combine(refs, "gpt_core_in.npy"));
        int t = (int)emb.Shape[1], d = (int)emb.Shape[2];
        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        using var g = new Graph(rt, dev);
        g.Enter();

        string p = "gpt.gpt.h.0.";
        var x = g.Input(new long[] { t, d }, emb.Data);
        var ln1 = Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-5f), w[p + "ln_1.weight"]), w[p + "ln_1.bias"]);
        var qkv = Ops.Add(Ops.Matmul(ln1, w[p + "attn.c_attn.weight"]), w[p + "attn.c_attn.bias"]);
        var mask = Ops.Cast(g.Input(new long[] { t, t }, Gpt2.CausalMask(t)), Ops.F16);
        var attnRaw = Gpt2.AttnRaw(w, 0, ln1, mask, t);
        // same split the port uses, kept separate so the heads can be inspected
        ulong rowBytes = (ulong)(3 * Gpt2.D * 4);
        var qH = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, Gpt2.D, t, rowBytes, 0)), t, Gpt2.Heads);
        var kH = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, Gpt2.D, t, rowBytes, (ulong)(Gpt2.D * 4))), t, Gpt2.Heads);
        var vH = Gpt2.ToHeads(Ops.Contiguous(Ops.View2d(qkv, Gpt2.D, t, rowBytes, (ulong)(2 * Gpt2.D * 4))), t, Gpt2.Heads);
        var attn = Ops.Add(Ops.Matmul(attnRaw, w[p + "attn.c_proj.weight"]), w[p + "attn.c_proj.bias"]);
        var m1 = Ops.Add(x, attn);
        var ln2 = Ops.Add(Ops.Mul(Ops.LayerNorm(m1, 1e-5f), w[p + "ln_2.weight"]), w[p + "ln_2.bias"]);
        var fc = Ops.Add(Ops.Matmul(ln2, w[p + "mlp.c_fc.weight"]), w[p + "mlp.c_fc.bias"]);
        var act = Ops.Gelu(fc);
        var mlp = Gpt2.Mlp(w, 0, ln2);

        Console.WriteLine($"  qkv.type={qkv.Type} (0=F32,1=F16)  qhead.type={qH.Type}");

        var stages = new (string Name, Tensor T)[] {
            ("ln1", ln1), ("qkv", qkv), ("attn_raw", attnRaw), ("attn", attn), ("m1", m1),
            ("ln2", ln2), ("fc", fc), ("act", act), ("mlp", mlp) };
        foreach (var (_, tt) in stages) tt.MarkOutput();
        qH.MarkOutput(); kH.MarkOutput(); vH.MarkOutput();
        var qV = qH.ToFloats(g); var kV = kH.ToFloats(g); var vV = vH.ToFloats(g);
        Report("gpt_l0_q", Npy.Load(Path.Combine(refs, "gpt_l0_q.npy")), qV);
        Report("gpt_l0_k", Npy.Load(Path.Combine(refs, "gpt_l0_k.npy")), kV);
        Report("gpt_l0_v", Npy.Load(Path.Combine(refs, "gpt_l0_v.npy")), vV);
        foreach (var (name, tt) in stages)
        {
            var v = tt.ToFloats(g);
            Report("gpt_l0_" + name, Npy.Load(Path.Combine(refs, $"gpt_l0_{name}.npy")), v);
            if (name == "attn_raw" || name == "qkv")
            {
                var bb = new byte[v.Length * 4];
                Buffer.BlockCopy(v, 0, bb, 0, bb.Length);
                File.WriteAllBytes(Path.Combine(refs, $"_cs_{name}.f32"), bb);
            }
        }
        g.Exit();
        return 0;
    }

    /// <summary>Validate the 24-layer GPT2 core + ln_f + final_norm + mel_head against PyTorch.</summary>
    private static int RunGpt2(Runtime rt, Device dev, string refs, string gguf)
    {
        var emb = Npy.Load(Path.Combine(refs, "gpt_core_in.npy"));               // [1,24,1280]
        int t = (int)emb.Shape[1];
        int d = (int)emb.Shape[2];
        Console.WriteLine($"inputs_embeds: t={t} d={d}");

        using var w = new GgufWeights(gguf, dev, new[] { "gpt." });
        Console.WriteLine($"gpt.* tensors: {w.Count}");

        using var g = new Graph(rt, dev);
        g.Enter();
        var x = g.Input(new long[] { t, d }, emb.Data);
        var (lastHidden, finalNorm) = Gpt2.Forward(w, g, x, t);
        var logits = Gpt2.MelHead(w, finalNorm);

        var lh = Ops.Contiguous(lastHidden).MarkOutput();
        var fn = Ops.Contiguous(finalNorm).MarkOutput();
        var lg = Ops.Contiguous(logits).MarkOutput();
        // first read computes the graph
        var lhV = lh.ToFloats(g);
        var fnV = fn.ToFloats(g);
        var lgV = lg.ToFloats(g);
        g.Exit();

        Report("last_hidden", Npy.Load(Path.Combine(refs, "gpt_core_last_hidden.npy")), lhV);
        Report("final_norm", Npy.Load(Path.Combine(refs, "gpt_final_norm_out.npy")), fnV);
        var wantLg = Npy.Load(Path.Combine(refs, "gpt_mel_logits.npy"));
        Report("mel_logits", wantLg, lgV);

        // argmax agreement is the metric that actually matters for sampling
        int agree = 0;
        for (int i = 0; i < t; i++)
        {
            int a = ArgMax(wantLg.Data, i * Gpt2.MelVocab, Gpt2.MelVocab);
            int b = ArgMax(lgV, i * Gpt2.MelVocab, Gpt2.MelVocab);
            if (a == b) agree++;
        }
        Console.WriteLine($"  argmax agreement: {agree}/{t}");
        return 0;
    }

    private static int ArgMax(float[] a, int off, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++) if (a[off + i] > a[off + best]) best = i;
        return best;
    }
}
