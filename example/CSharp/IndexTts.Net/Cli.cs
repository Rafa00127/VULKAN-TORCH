using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NAudio.Wave;
using IndexTtsSharp;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 command line: reference wav + text -> wav, thin wrapper over the
/// <see cref="IndexTtsSharp.Tts"/> library (kept for build/validation/demo).
///
///   synth (default)   --ref-wav r.wav --text "..." -> --out y.wav
///
/// Reference-audio chain: Kaldi fbank -> CAMPPlus (style); SeamlessM4T features -> Wav2Vec2Bert
/// (speaker conditioning); 22.05 kHz mel (ref_mel). Then per text segment: GPT-2 AR -> semantic
/// codec -> length regulator -> CFM (25-step Euler, CFG) -> BigVGAN.
/// </summary>
internal static class Cli
{
    public static int Run(string[] args)
    {
        if (IsHelp(args))
        {
            PrintHelp();
            return 0;
        }

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

        var sw = Stopwatch.StartNew();
        using var tts = new Tts(model, tokenizer);
        double wLoadMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"backend: {tts.BackendName}");
        Console.WriteLine($"weights: {tts.TensorCount} tensors in {wLoadMs:F0} ms");

        // ---- text front-end + reference audio (one-time prep) -----------------------------
        var prepSw = Stopwatch.StartNew();
        var wav = LoadWav(refWav, out int sr);
        var voice = tts.EncodeReference(wav, sr);
        double prepMs = prepSw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"ref_mel: {voice.RefFrames} frames   (prep {prepMs:F0} ms)");

        var opts = new IndexOptions
        {
            Lang = lang, Emotion = emoSpec, MaxSteps = maxSteps, Seed = seed,
            TopK = topK, TopP = topP, Temperature = temp, RepPenalty = repPen,
            CfgRate = cfgRate, DiffusionSteps = diffusionSteps, DurationFactor = durationFactor,
            NumBeams = numBeams, UseReplay = useReplay,
        };
        if (emoSpec != null) Console.WriteLine($"emotion: {emoSpec}");

        var arSw = Stopwatch.StartNew();
        var codes = tts.GenerateCodes(text, voice, opts);
        double arMs = arSw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"GPT-2 AR: {codes.Count(c => c >= 0)} mel codes ({arMs:F0} ms)");

        double s2melMs = 0, vganMs = 0;
        var decSw = Stopwatch.StartNew();
        var pcm = tts.Decode(codes, voice, opts, (s2m, vg) => { s2melMs = s2m; vganMs = vg; });
        double decMs = decSw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Decode: s2mel {s2melMs:F0} ms + BigVGAN {vganMs:F0} ms ({decMs:F0} ms)");

        Program.SaveWav(outPath, pcm, Tts.OutputSampleRate);
        double audioSec = pcm.Length / (double)Tts.OutputSampleRate;
        double synthMs = prepMs + arMs + s2melMs + vganMs;   // excludes one-off weight load, like the reference's timer
        Console.WriteLine();
        Console.WriteLine("=== Timing ===");
        Console.WriteLine($"Weights:           {wLoadMs,8:F0} ms");
        Console.WriteLine($"Prep (text+ref):   {prepMs,8:F0} ms");
        Console.WriteLine($"GPT-2 AR:          {arMs,8:F0} ms");
        Console.WriteLine($"s2mel (codec+CFM): {s2melMs,8:F0} ms");
        Console.WriteLine($"BigVGAN:           {vganMs,8:F0} ms");
        Console.WriteLine($"Inference total:   {synthMs,8:F0} ms");
        Console.WriteLine($"Audio:             {audioSec,8:F2} sec");
        Console.WriteLine($"RTF:               {synthMs / 1000.0 / audioSec,8:F3} x");
        Console.WriteLine($"Wall (incl. load): {sw.Elapsed.TotalMilliseconds,8:F0} ms");
        Console.WriteLine($"wrote {outPath}");
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
@"IndexTTS 2.5 CLI - reference wav + text -> wav (zero-shot voice clone + emotion control)

Usage:
  IndexTts.Net.exe synth --model <gguf> --ref-wav <wav> --text ""..."" [options]

Options:
  --model <gguf>       IndexTTS 2.5 GGUF (required; default model/indextts2.5/indextts2.5.f16.gguf)
  --tokenizer <path>   tiktoken vocab (default model/indextts2.5/multilingual_zh_ja_yue_char_del.tiktoken)
  --ref-wav <wav>      reference audio, the timbre source (default data/ref_audio/melinaref_24k.wav)
  --text <text>        text to synthesize; supports <char|pronunciation> annotations
  --lang <code>        language: zh (default), en, ja, es, ... (99 languages)
  --emo <spec>         emotion weights, e.g. ""happy=0.6,calm=0.4""
                       (happy/angry/sad/afraid/disgusted/melancholic/surprised/calm)
  --out <path>         output wav (default data/indextts/cli.wav)
  --seed <n>           RNG seed (default 42)
  --max-steps <n>      AR step cap; 0 = max(64, len(text)*8) (default 0)
  --num-beams <n>      beam-search width (default 1)
  --top-p <f>          nucleus sampling (default 0.8)
  --top-k <n>          top-k sampling (default 30)
  --temperature <f>    sampling temperature (default 0.8)
  --rep-penalty <f>    repetition penalty (default 10)
  --cfg-rate <f>       CFM classifier-free-guidance rate (default 0.7)
  --diffusion-steps <n> CFM Euler steps (default 25)
  --duration-factor <f> speech-rate / duration scale (default 1.0)
  --no-graph-cache     rebuild the AR graph every step (A/B against the cached path)
  -h, --help           show this help

Example:
  IndexTts.Net.exe synth --model model\indextts2.5\indextts2.5.f16.gguf ^
    --ref-wav data\ref_audio\melinaref_24k.wav ^
    --text ""大家好，这是一个测试。"" --out data\indextts\out.wav");
    }

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
}
