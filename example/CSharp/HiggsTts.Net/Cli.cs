using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using VulkanTorch;

namespace HiggsTts;

/// <summary>
/// HiggsTTS command-line, self-contained (like higgstts_py/cli.py):
/// reference wav + text -> wav, no Python-side export needed.
///
///   synth   (default) ref wav + text -> wav     [encode_ref + AR + DAC decode]
///   encode            ref wav -> RVQ codes      [encode_ref only]
/// </summary>
internal static class Cli
{
    // prompt special token ids (from higgs_tts.h / higgstts_py/tts.py)
    private const int TokTts = 151667, TokRefText = 151680, TokRefAudio = 151679;
    private const int TokText = 151672, TokAudio = 151670, AudioPlaceholder = -100;

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

    private static int[] BuildPrompt(HiggsTokenizer tok, string text, string refText, int numRef)
    {
        var p = new List<int> { TokTts };
        if (!string.IsNullOrEmpty(refText))
        {
            p.Add(TokRefText);
            p.AddRange(tok.Encode(refText));
        }
        p.Add(TokRefAudio);
        for (int i = 0; i < numRef; i++) p.Add(AudioPlaceholder);
        p.Add(TokText);
        if (!string.IsNullOrEmpty(text)) p.AddRange(tok.Encode(text));
        p.Add(TokAudio);
        return p.ToArray();
    }

    public static int Run(string[] args)
    {
        string mode = "synth", refText = "", text = "";
        string? model = null, refWav = null, outPath = null, tokenizer = null;
        float temperature = 0.9f;
        int topk = 50, seed = 42, maxSteps = 0;   // 0 = predict from text length
        bool noCache = false;

        var positional = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model": model = args[++i]; break;
                case "--ref-wav": refWav = args[++i]; break;
                case "--ref-text": refText = args[++i]; break;
                case "--text": text = args[++i]; break;
                case "--out": outPath = args[++i]; break;
                case "--tokenizer": tokenizer = args[++i]; break;
                case "--temperature": temperature = float.Parse(args[++i]); break;
                case "--topk": topk = int.Parse(args[++i]); break;
                case "--seed": seed = int.Parse(args[++i]); break;
                case "--max-steps": maxSteps = int.Parse(args[++i]); break;
                case "--no-graph-cache": noCache = true; break;
                default: positional.Add(args[i]); break;
            }
        }
        if (positional.Count > 0) mode = positional[0];
        if (model == null)
        {
            Console.Error.WriteLine("cli: --model <higgs gguf> is required");
            return 2;
        }

        string root = Program.FindRoot();
        string data = Path.Combine(root, "data", "ref_audio");
        refWav ??= Path.Combine(data, "melinaref_24k.wav");
        tokenizer ??= Path.Combine(data, "higgs_tts_v3_tokenizer.json");
        outPath ??= Path.Combine(root, "data", "higgstts", $"cli_cs_{mode}.wav");

        using var rt = new Runtime();
        var dev = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");

        // ---- load the prefill (codec) weights; + backbone for synth ----
        var sw = Stopwatch.StartNew();
        string[] prefixes = mode == "encode"
            ? HiggsWeights.PrefillPrefixes
            : HiggsWeights.PrefillPrefixes.Concat(HiggsWeights.BackbonePrefixes).ToArray();
        using var w = new HiggsWeights(model, dev, prefixes, 4UL << 30);
        EncodeRef.BuildPceWeight(w);
        Console.WriteLine($"load weights:        {sw.Elapsed.TotalMilliseconds,8:F1} ms  ({w.Count} tensors)");

        var wav = LoadWav(refWav, out int sr);
        sw.Restart();
        var refOut = EncodeRef.Run(rt, dev, w, wav, sr);
        double encMs = sw.Elapsed.TotalMilliseconds;
        int rows = refOut.GetLength(0);
        Console.WriteLine($"Prefill: {rows} frames x {Ar.NCb} codebooks ({encMs:F0} ms)");
        Console.WriteLine($"  first: {string.Join(", ", Enumerable.Range(0, Ar.NCb).Select(c => refOut[0, c]))}");

        if (mode == "encode")
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
            var flat = new int[rows * Ar.NCb];
            for (int i = 0; i < rows; i++)
                for (int c = 0; c < Ar.NCb; c++) flat[i * Ar.NCb + c] = refOut[i, c];
            var bytes = new byte[flat.Length * 4];
            Buffer.BlockCopy(flat, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(outPath + ".i32", bytes);
            Console.WriteLine($"wrote {outPath}.i32");
            return 0;
        }

        // ---- synth: tokenize, build prompt, AR, decode ----
        var tok = new HiggsTokenizer(tokenizer);
        var ids = tok.Encode(text);
        Console.WriteLine($"text tokens: {ids.Length} -> [{string.Join(", ", ids)}]");

        var refFlat = new int[rows * Ar.NCb];
        for (int i = 0; i < rows; i++)
            for (int c = 0; c < Ar.NCb; c++) refFlat[i * Ar.NCb + c] = refOut[i, c];

        var prompt = BuildPrompt(tok, text, refText, rows + Ar.NCb - 1);
        Console.WriteLine($"prompt L={prompt.Length}  ref_codes=[{rows}, {Ar.NCb}]");

        sw.Restart();
        var codes = Ar.Generate(rt, dev, w, prompt, refFlat, rows, out int steps,
                                temperature, seed, maxSteps, topk, graphCache: !noCache);
        double arMs = sw.Elapsed.TotalMilliseconds;
        int genRows = codes.Length / Ar.NCb;
        Console.WriteLine($"Backbone AR: {steps} raw frames ({arMs:F0} ms)");

        sw.Restart();
        var pcm = DacDecoder.Decode(rt, dev, w, codes, genRows);
        double decMs = sw.Elapsed.TotalMilliseconds;
        double dur = pcm.Length / 24000.0;
        double totalMs = encMs + arMs + decMs;
        Console.WriteLine($"Decode: {pcm.Length} PCM samples ({dur:F2} sec) ({decMs:F0} ms)");

        // same summary block as the reference higgs_cli.exe
        Console.WriteLine();
        Console.WriteLine("=== Timing ===");
        Console.WriteLine($"Prefill:      {encMs,8:F0} ms");
        Console.WriteLine($"Backbone AR:  {arMs,8:F0} ms");
        Console.WriteLine($"Decode:       {decMs,8:F0} ms");
        Console.WriteLine($"Total:        {totalMs,8:F0} ms");
        Console.WriteLine($"Audio:        {dur,8:F2} sec");
        Console.WriteLine($"RTF:          {totalMs / 1000.0 / dur,8:F3} x");

        Program.SaveWav(outPath, pcm);
        Console.WriteLine($"\nSaved: {outPath}");
        return 0;
    }
}
