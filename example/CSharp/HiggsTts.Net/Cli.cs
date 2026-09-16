using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NAudio.Wave;
using HiggsTtsSharp;

namespace HiggsTts;

/// <summary>
/// HiggsTTS command-line: reference wav + text -> wav, thin wrapper over the
/// <see cref="HiggsTtsSharp.Tts"/> library (kept for build/validation/demo).
///
///   synth   (default) ref wav + text -> wav     [encode_ref + AR + DAC decode]
///   encode            ref wav -> RVQ codes      [encode_ref only]
/// </summary>
internal static class Cli
{
    /// <summary>Reference audio -> mono float samples.
    ///
    /// WAV goes straight to NAudio.Core's managed reader; anything else is piped through
    /// ffmpeg (must be on PATH) and decoded to WAV in memory -- no temp file. Using
    /// WaveFileReader instead of AudioFileReader keeps this CLI off the Windows-only
    /// NAudio metapackage.</summary>
    private static float[] LoadWav(string path, out int sampleRate)
    {
        using var src = IsRiff(path) ? File.OpenRead(path) : FfmpegWav(path);
        using var r = new WaveFileReader(src);
        var sp = r.ToSampleProvider();
        sampleRate = r.WaveFormat.SampleRate;
        int ch = r.WaveFormat.Channels;
        var buf = new List<float>();
        var tmp = new float[4096];
        int n;
        while ((n = sp.Read(tmp, 0, tmp.Length)) > 0)
            for (int i = 0; i < n; i++) buf.Add(tmp[i]);
        var all = buf.ToArray();
        if (ch == 1) return all;
        var mono = new float[all.Length / ch];
        for (int i = 0; i < mono.Length; i++) mono[i] = all[i * ch];
        return mono;
    }

    /// <summary>Sniff the container, not the extension: "RIFF" at offset 0 = WAV.</summary>
    private static bool IsRiff(string path)
    {
        using var f = File.OpenRead(path);
        var magic = new byte[4];
        return f.Read(magic, 0, 4) == 4 && magic[0] == 'R' && magic[1] == 'I'
            && magic[2] == 'F' && magic[3] == 'F';
    }

    /// <summary>ffmpeg -i &lt;path&gt; -f wav - : decode to WAV on stdout, buffered in memory
    /// (WaveFileReader needs to seek). ffmpeg's stderr stays attached to the console.</summary>
    private static Stream FfmpegWav(string path)
    {
        var psi = new ProcessStartInfo("ffmpeg")
        {
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        foreach (var a in new[] { "-v", "error", "-i", path, "-f", "wav", "-" })
            psi.ArgumentList.Add(a);
        Process p;
        try
        {
            p = Process.Start(psi)!;
        }
        catch (Exception e)
        {
            throw new InvalidOperationException(
                $"'{path}' is not a WAV and ffmpeg could not be started: {e.Message}");
        }
        var mem = new MemoryStream();
        p.StandardOutput.BaseStream.CopyTo(mem);
        p.WaitForExit();
        if (p.ExitCode != 0)
            throw new InvalidOperationException($"ffmpeg failed on '{path}' (exit {p.ExitCode})");
        return new MemoryStream(FixPipedSizes(mem));
    }

    /// <summary>ffmpeg cannot seek a pipe, so it writes 0xFFFFFFFF placeholders where the
    /// RIFF chunk and the trailing "data" chunk expect a size. Fill them in from the real
    /// length, or NAudio's strict RIFF reader rejects the buffer.</summary>
    private static byte[] FixPipedSizes(MemoryStream ms)
    {
        var b = ms.ToArray();
        void PutU32(int at, long v)
        {
            for (int k = 0; k < 4; k++) b[at + k] = (byte)(v >> (8 * k));
        }
        PutU32(4, b.Length - 8);
        for (int i = 12; i + 8 <= b.Length;)
        {
            if (b[i] == 'd' && b[i + 1] == 'a' && b[i + 2] == 't' && b[i + 3] == 'a')
            {
                PutU32(i + 4, b.Length - i - 8);
                break;
            }
            long sz = BitConverter.ToUInt32(b, i + 4);
            if (sz == 0xFFFFFFFF || sz > 0x7FFFFF00)
                break;                                       // cannot walk further safely
            i += 8 + (int)sz;
        }
        return b;
    }

    public static int Run(string[] args)
    {
        if (IsHelp(args)) { PrintHelp(); return 0; }

        string mode = "synth", refText = "", text = "";
        string? model = null, refWav = null, outPath = null, tokenizer = null, codesFile = null, f32Out = null;
        float temperature = 0.9f;
        int topk = 50, seed = 42, maxSteps = 0;   // 0 = predict from text length
        bool noCache = false;

        var positional = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model": model = args[++i]; break;
                case "--codes": codesFile = args[++i]; break;
                case "--f32": f32Out = args[++i]; break;
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
        // tokenizer stays null by default → the library builds it from the GGUF's vocab+merges.
        outPath ??= Path.Combine(root, "data", "higgstts", $"cli_cs_{mode}.wav");

        var sw = Stopwatch.StartNew();
        using var tts = new Tts(model, tokenizer);
        Console.WriteLine($"backend: {tts.BackendName}");
        Console.WriteLine($"load weights:        {sw.Elapsed.TotalMilliseconds,8:F1} ms  ({tts.TensorCount} tensors)");

        // `decode` takes a codes file straight to PCM, with no encode_ref / AR in the way —
        // the counterpart of `encode`, and the way to A/B this decoder against another port's
        // on identical input (tools/check_higgs_ref.py --wav writes the Python side).
        if (mode == "decode")
        {
            if (codesFile == null)
            {
                Console.Error.WriteLine("decode: --codes <file.i32> is required (flat int32 [T*8])");
                return 2;
            }
            var raw = File.ReadAllBytes(codesFile);
            var codesIn = new int[raw.Length / sizeof(int)];
            Buffer.BlockCopy(raw, 0, codesIn, 0, raw.Length);
            sw.Restart();
            var dec = tts.Decode(codesIn);
            Console.WriteLine($"Decode: {dec.Length} PCM samples "
                              + $"({dec.Length / (double)Tts.SampleRate:F2} sec) "
                              + $"({sw.Elapsed.TotalMilliseconds:F0} ms)");
            if (f32Out != null)
            {
                // 16-bit wavs can't be compared byte-for-byte across ports (the writers
                // round differently); this is the raw output for numeric A/B.
                var fb = new byte[dec.Length * 4];
                Buffer.BlockCopy(dec, 0, fb, 0, fb.Length);
                File.WriteAllBytes(f32Out, fb);
                Console.WriteLine($"wrote raw f32 PCM -> {f32Out}");
            }
            Program.SaveWav(outPath, dec);
            Console.WriteLine($"\nSaved: {outPath}");
            return 0;
        }

        var wav = LoadWav(refWav, out int sr);
        sw.Restart();
        var voice = tts.EncodeReference(wav, sr, refText);
        double encMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Prefill: {voice.Rows} frames x 8 codebooks ({encMs:F0} ms)");
        Console.WriteLine($"  first: {string.Join(", ", voice.Codes.Take(8))}");

        if (mode == "encode")
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
            var bytes = new byte[voice.Codes.Length * 4];
            Buffer.BlockCopy(voice.Codes, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(outPath + ".i32", bytes);
            Console.WriteLine($"wrote {outPath}.i32");
            return 0;
        }

        var ids = tts.EncodeText(text);
        Console.WriteLine($"text tokens: {ids.Length} -> [{string.Join(", ", ids)}]");
        Console.WriteLine($"prompt ref_codes=[{voice.Rows}, 8]");

        var opts = new HiggsOptions
        {
            Temperature = temperature, TopK = topk, Seed = seed,
            MaxSteps = maxSteps, GraphCache = !noCache,
        };
        sw.Restart();
        var codes = tts.GenerateCodes(text, voice, opts);
        double arMs = sw.Elapsed.TotalMilliseconds;
        int genRows = codes.Length / 8;
        Console.WriteLine($"Backbone AR: {genRows} raw frames ({arMs:F0} ms)");

        sw.Restart();
        var pcm = tts.Decode(codes);
        double decMs = sw.Elapsed.TotalMilliseconds;
        double dur = pcm.Length / (double)Tts.SampleRate;
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

    private static bool IsHelp(string[] args)
    {
        if (args.Length == 0) return false;
        if (args[0] is "-h" or "--help" or "help") return true;
        return args.Any(a => a is "-h" or "--help");
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
@"HiggsTTS CLI - reference audio + text -> wav (encode_ref + Qwen3 AR + DAC decode)

Usage:
  HiggsTts.Net.exe [synth] --model <gguf> [options]
  HiggsTts.Net.exe encode  --model <gguf> [--ref-wav <wav>] [--out <path>]
  HiggsTts.Net.exe decode  --model <gguf> --codes <file.i32> [--out <path>]

Modes:
  synth      (default) reference wav + text -> wav
  encode     reference wav -> RVQ codes (writes <out>.i32); stops after encode_ref
  decode     RVQ codes -> wav; skips encode_ref and the AR entirely

Options:
  --model <gguf>       HiggsTTS GGUF (required)
  --codes <file>       `decode` input: flat int32, [T*8] row-major (what `encode` writes)
  --tokenizer <json>   override the BPE tokenizer with an HF tokenizer.json
                       (default: rebuilt from the GGUF's embedded vocab + merges)
  --ref-wav <wav>      reference audio (default: data/ref_audio/melinaref_24k.wav)
  --ref-text <text>    transcript of the reference audio (optional but recommended)
  --text <text>        text to synthesize; supports <|style:whispering|> tags
  --out <path>         output wav (default: data/higgstts/cli_cs_<mode>.wav)
  --temperature <f>    sampling temperature (default 0.9)
  --topk <n>           top-k sampling (default 50)
  --seed <n>           RNG seed (default 42)
  --max-steps <n>      AR step budget; 0 = predict from text length (default 0)
  --no-graph-cache     rebuild the AR graph every step (A/B against the cached path)
  -h, --help           show this help

Examples:
  HiggsTts.Net.exe synth --model D:\models\HiggsTTS3.gguf ^
    --ref-text ""I have no doubt you will become Elden Lord."" ^
    --text ""<|style:whispering|>Hello, how are you?""");
    }
}
