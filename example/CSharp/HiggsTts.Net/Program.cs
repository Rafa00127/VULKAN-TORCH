using System;
using System.Diagnostics;
using System.IO;
using HiggsTts;
using Minitorch;
using NAudio.Wave;

internal static class Program
{
    private const string DefaultGguf =
        @"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf";

    internal static string FindRoot()
    {
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            if (File.Exists(Path.Combine(d.FullName, "build.py")))
                return d.FullName;
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("repo root (build.py) not found above the exe");
    }

    private static int[] ReadInts(string path)
    {
        var raw = File.ReadAllBytes(path);
        var v = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
        return v;
    }

    private static float[] ReadFloats(string path)
    {
        var raw = File.ReadAllBytes(path);
        var v = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
        return v;
    }

    private static void Report(string tag, float[] want, float[] got)
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
        double snr = se == 0 ? double.PositiveInfinity : 10 * Math.Log10(sRef / se);
        string snrS = double.IsInfinity(snr) ? "exact" : $"{snr:F1} dB";
        Console.WriteLine($"  [{tag}] n={n} max_err={maxErr:E3} snr={snrS}");
    }

    private static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "cli")
            return Cli.Run(args[1..]);

        string mode = args.Length > 0 ? args[0] : "decode";
        string gguf = args.Length > 1 ? args[1] : DefaultGguf;
        string root = FindRoot();
        string refs = Path.Combine(root, "data", "higgstts");

        using var rt = new Runtime();
        var dev = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");

        return mode switch
        {
            "ar" => RunAr(rt, dev, refs, gguf),
            "gen" => RunGen(rt, dev, refs, gguf),
            _ => RunDecode(rt, dev, refs, gguf),
        };
    }

    // ---- real generation: prompt + ref codes -> AR codes -> wav ----
    private static int RunGen(Runtime rt, Device dev, string refs, string gguf)
    {
        var promptIds = ReadInts(Path.Combine(refs, "ar_gen_prompt_ids.i32"));
        var refCodes = ReadInts(Path.Combine(refs, "ar_gen_ref_codes.i32"));
        int refRows = refCodes.Length / Ar.NCb;
        Console.WriteLine($"prompt L={promptIds.Length} ref_codes=[{refRows}, {Ar.NCb}]");

        var sw = Stopwatch.StartNew();
        using var wb = new HiggsWeights(gguf, dev, HiggsWeights.BackbonePrefixes, 4UL << 30);
        Console.WriteLine($"backbone: {wb.Count} tensors in {sw.ElapsedMilliseconds} ms");

        sw.Restart();
        var codes = Ar.Generate(rt, dev, wb, promptIds, refCodes, refRows, out int steps, 0.9f, 42, 400, 50);
        int rows = codes.Length / Ar.NCb;
        Console.WriteLine($"generated [{rows}, {Ar.NCb}] codes in {sw.ElapsedMilliseconds} ms");

        var codeBytes = new byte[codes.Length * 4];
        Buffer.BlockCopy(codes, 0, codeBytes, 0, codeBytes.Length);
        File.WriteAllBytes(Path.Combine(refs, "gen_cs_codes.i32"), codeBytes);

        sw.Restart();
        using var wd = new HiggsWeights(gguf, dev, HiggsWeights.DecodePrefixes);
        var pcm = DacDecoder.Decode(rt, dev, wd, codes, rows);
        Console.WriteLine($"decode: {pcm.Length} samples in {sw.ElapsedMilliseconds} ms");

        string outWav = Path.Combine(refs, "gen_cs.wav");
        SaveWav(outWav, pcm);
        Console.WriteLine($"wrote {outWav} ({pcm.Length / 24000.0:F2}s)");
        return 0;
    }

    // ---- AR backbone: prefill + teacher-forced decode steps vs Python logits ----
    private static int RunAr(Runtime rt, Device dev, string refs, string gguf)
    {
        var promptIds = ReadInts(Path.Combine(refs, "ar_prompt_ids.i32"));
        var delayed = ReadInts(Path.Combine(refs, "ar_delayed_codes.i32"));
        var stepCodes = ReadInts(Path.Combine(refs, "ar_step_codes.i32"));
        int l = promptIds.Length;
        int la = delayed.Length / Ar.NCb;
        int steps = stepCodes.Length / Ar.NCb;
        int maxCtx = l + 16;
        Console.WriteLine($"prompt L={l} La={la} steps={steps} max_ctx={maxCtx}");

        var sw = Stopwatch.StartNew();
        using var w = new HiggsWeights(gguf, dev, HiggsWeights.BackbonePrefixes, 4UL << 30);
        Console.WriteLine($"weights: {w.Count} tensors in {sw.ElapsedMilliseconds} ms");

        using var kv = new Ar.KVCache(dev, maxCtx);

        sw.Restart();
        var pre = Ar.PrefillLogits(rt, dev, w, kv, promptIds, delayed, la);
        Console.WriteLine($"prefill: {pre.Length} logits in {sw.ElapsedMilliseconds} ms");
        Report("prefill", ReadFloats(Path.Combine(refs, "ar_prefill_logits.f32")), pre);

        var stepRef = ReadFloats(Path.Combine(refs, "ar_step_logits.f32"));
        int per = Ar.NCb * Ar.CbVocab;
        for (int i = 0; i < steps; i++)
        {
            var cn = new int[Ar.NCb];
            Array.Copy(stepCodes, i * Ar.NCb, cn, 0, Ar.NCb);
            var lg = Ar.StepLogits(rt, dev, w, kv, cn, l + i);
            var want = new float[per];
            Array.Copy(stepRef, i * per, want, 0, per);
            Report($"step {i}", want, lg);
        }
        return 0;
    }

    // ---- DAC decoder: codes -> PCM, vs Python ----
    private static int RunDecode(Runtime rt, Device dev, string refs, string gguf)
    {
        var codes = ReadInts(Path.Combine(refs, "dec_codes.i32"));
        int frames = codes.Length / 8;
        Console.WriteLine($"codes: [{frames}, 8]");

        var sw = Stopwatch.StartNew();
        using var w = new HiggsWeights(gguf, dev, HiggsWeights.DecodePrefixes);
        Console.WriteLine($"weights: {w.Count} tensors in {sw.ElapsedMilliseconds} ms");

        sw.Restart();
        var pcm = DacDecoder.Decode(rt, dev, w, codes, frames);
        Console.WriteLine($"decode: {pcm.Length} samples in {sw.ElapsedMilliseconds} ms");

        Report("decode", ReadFloats(Path.Combine(refs, "dec_py.f32")), pcm);

        string outWav = Path.Combine(refs, "dec_cs.wav");
        SaveWav(outWav, pcm);
        Console.WriteLine($"wrote {outWav} ({pcm.Length / 24000.0:F2}s)");
        return 0;
    }

    internal static void SaveWav(string path, float[] s, int sampleRate = 24000)
    {
        using var w = new WaveFileWriter(path, new WaveFormat(sampleRate, 16, 1));
        var bytes = new byte[s.Length * 2];
        for (int i = 0; i < s.Length; i++)
        {
            short pcm = (short)Math.Round(Math.Clamp(s[i], -1f, 1f) * 32767f);
            bytes[2 * i] = (byte)pcm;
            bytes[2 * i + 1] = (byte)(pcm >> 8);
        }
        w.Write(bytes, 0, bytes.Length);
    }
}
