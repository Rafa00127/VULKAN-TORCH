using System;
using System.Diagnostics;
using System.IO;
using Minitorch;

namespace HiggsTts;

/// <summary>
/// HiggsTTS command-line (AR backbone + DAC decode). Mirrors higgstts_py/cli.py;
/// both take a pre-encoded prompt so the two implementations run the same work.
/// </summary>
internal static class Cli
{
    private const string DefaultGguf =
        @"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf";

    private static int[] ReadInts(string path)
    {
        var raw = File.ReadAllBytes(path);
        var v = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
        return v;
    }

    public static int Run(string[] args)
    {
        string gguf = DefaultGguf;
        string? promptIds = null, refCodes = null, outPath = null;
        float temperature = 0.9f;
        int topk = 50, seed = 42, maxSteps = 400;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--gguf": gguf = args[++i]; break;
                case "--prompt-ids": promptIds = args[++i]; break;
                case "--ref-codes": refCodes = args[++i]; break;
                case "--out": outPath = args[++i]; break;
                case "--temperature": temperature = float.Parse(args[++i]); break;
                case "--topk": topk = int.Parse(args[++i]); break;
                case "--seed": seed = int.Parse(args[++i]); break;
                case "--max-steps": maxSteps = int.Parse(args[++i]); break;
                default:
                    Console.Error.WriteLine($"cli: unknown argument '{args[i]}'");
                    return 2;
            }
        }
        if (promptIds == null || refCodes == null)
        {
            Console.Error.WriteLine("cli: --prompt-ids and --ref-codes are required");
            return 2;
        }
        outPath ??= Path.Combine(Program.FindRoot(), "data", "higgstts", "cli_cs.wav");

        using var rt = new Runtime();
        var dev = rt.Gpu();

        var sw = Stopwatch.StartNew();
        using var w = new HiggsWeights(gguf, dev, HiggsWeights.BackbonePrefixes, 4UL << 30);
        Console.WriteLine($"load weights:        {sw.Elapsed.TotalMilliseconds,8:F1} ms  "
                          + $"({w.Count} tensors, {rt.Name})");

        var prompt = ReadInts(promptIds);
        var refC = ReadInts(refCodes);
        int refRows = refC.Length / Ar.NCb;

        sw.Restart();
        var codes = Ar.Generate(rt, dev, w, prompt, refC, refRows, out int steps,
                                temperature, seed, maxSteps, topk);
        double arMs = sw.Elapsed.TotalMilliseconds;
        int rows = codes.Length / Ar.NCb;

        Console.WriteLine($"prompt L={prompt.Length}  ref_codes=[{refRows}, {Ar.NCb}]  "
                          + $"->  codes [{rows}, {Ar.NCb}]");
        Console.WriteLine($"AR (prefill+{steps} steps): {arMs,8:F1} ms  ({arMs / steps,6:F2} ms/step)");

        using var wd = new HiggsWeights(gguf, dev, HiggsWeights.DecodePrefixes);
        sw.Restart();
        var pcm = DacDecoder.Decode(rt, dev, wd, codes, rows);
        double decMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine($"DAC decode:          {decMs,8:F1} ms");
        Console.WriteLine($"total (excl. load):  {arMs + decMs,8:F1} ms");
        double dur = pcm.Length / 24000.0;
        Console.WriteLine($"audio:               {pcm.Length} samples = {dur:F2} s "
                          + $"({(arMs + decMs) / 1e3 / dur:F2}x realtime)");

        Program.SaveWav(outPath, pcm);
        Console.WriteLine($"wrote {outPath}");
        return 0;
    }
}
