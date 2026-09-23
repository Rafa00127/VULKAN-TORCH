using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HiggsTts;
using VulkanTorch;

namespace HiggsTtsSharp;

/// <summary>Synthesis options (mirror the HiggsTTS CLI / higgstts_vt defaults).</summary>
public sealed class HiggsOptions
{
    public float Temperature { get; set; } = 0.9f;
    public int TopK { get; set; } = 50;
    public int Seed { get; set; } = 42;
    /// <summary>AR step budget; 0 = predict from text length (12 * text tokens + 200).</summary>
    public int MaxSteps { get; set; }
    /// <summary>Replay the captured decode graph instead of rebuilding it each step.</summary>
    public bool GraphCache { get; set; } = true;
    /// <summary>Optional style tag (e.g. <c>&lt;|style:whispering|&gt;</c>) prepended to the text.</summary>
    public string? StyleTag { get; set; }

    /// <summary>Streaming: frames of context each decode window carries on both sides of
    /// what it emits. Latency vs. closeness to a one-shot decode — see
    /// <see cref="DacStream.DefaultLookahead"/>.</summary>
    public int StreamLookaheadFrames { get; set; } = DacStream.DefaultLookahead;
    /// <summary>Streaming: frames per emitted chunk. 8 → 320 ms of audio per chunk.</summary>
    public int StreamStepFrames { get; set; } = 8;
    /// <summary>Streaming: samples of trailing silence that end the stream early. 0 or
    /// negative disables the rule (used when comparing against a one-shot decode, which
    /// has no such cut-off).</summary>
    public int SilenceStopSamples { get; set; } = 4 * Tts.SampleRate;
}

/// <summary>
/// HiggsTTS synthesis facade: reference audio → <see cref="ReferenceVoice"/>, then
/// text + voice → 24 kHz PCM. Self-contained (owns its Runtime / device / weights).
/// Not thread-safe — serialize calls on one instance.
/// </summary>
public sealed class Tts : IDisposable
{
    public const int SampleRate = 24000;

    // prompt special token ids (from higgs_tts.h / higgstts_vt/tts.py)
    private const int TokTts = 151667, TokRefText = 151680, TokRefAudio = 151679;
    private const int TokText = 151672, TokAudio = 151670, AudioPlaceholder = -100;

    private readonly Runtime _rt;
    private readonly Device _dev;
    private readonly HiggsWeights _w;
    private HiggsTokenizer? _tok;
    private bool _disposed;

    public Tts(string modelGgufPath, string? tokenizerJsonPath = null)
    {
        _rt = new Runtime();
        _dev = _rt.Gpu();
        _w = new HiggsWeights(modelGgufPath, _dev,
            HiggsTts.HiggsWeights.PrefillPrefixes.Concat(HiggsTts.HiggsWeights.BackbonePrefixes).ToArray(),
            4UL << 30);
        EncodeRef.BuildPceWeight(_w);
        // The GGUF carries the vocab+merges, so a separate tokenizer.json is optional.
        _tok = !string.IsNullOrWhiteSpace(tokenizerJsonPath)
            ? new HiggsTokenizer(tokenizerJsonPath)
            : HiggsTokenizer.FromGguf(modelGgufPath);
    }

    /// <summary>Override the BPE tokenizer with an HF tokenizer.json (empty path = keep the GGUF-built one).</summary>
    public bool SetTokenizer(string tokenizerJsonPath)
    {
        if (string.IsNullOrWhiteSpace(tokenizerJsonPath)) return true;
        _tok = new HiggsTokenizer(tokenizerJsonPath);
        return true;
    }

    public bool HasTokenizer => _tok != null;

    /// <summary>Encode reference audio (any sample rate) → a reusable <see cref="ReferenceVoice"/>.
    ///
    /// This is the one part of a request that does not depend on the text, and it is not
    /// cheap (~125 ms for a 15 s reference), so a server should call <see cref="Warmup"/>
    /// once per voice at startup instead of letting the first request pay for it.</summary>
    public ReferenceVoice EncodeReference(float[] audio, int sampleRate, string? refText = null)
    {
        var codes = EncodeRef.Run(_rt, _dev, _w, audio, sampleRate);   // int[T, 8]; resamples internally
        int t = codes.GetLength(0);
        var flat = new int[t * Ar.NCb];
        for (int i = 0; i < t; i++)
            for (int c = 0; c < Ar.NCb; c++) flat[i * Ar.NCb + c] = codes[i, c];
        return new ReferenceVoice(flat, refText);
    }

    /// <summary>Text + voice → raw RVQ codes (flat [T*8], row-major).</summary>
    public int[] GenerateCodes(string text, ReferenceVoice voice, HiggsOptions? options = null)
    {
        options ??= new HiggsOptions();
        if (_tok == null) throw new InvalidOperationException("no tokenizer; call SetTokenizer first");
        if (!string.IsNullOrEmpty(options.StyleTag)) text = options.StyleTag + text;

        var prompt = BuildPrompt(_tok, text, voice.RefText, voice.Rows + Ar.NCb - 1);
        return Ar.Generate(_rt, _dev, _w, prompt, voice.Codes, voice.Rows, out _,
                           options.Temperature, options.Seed, options.MaxSteps, options.TopK, options.GraphCache);
    }

    /// <summary>Raw RVQ codes → 24 kHz float32 PCM.</summary>
    public float[] Decode(int[] codes) => DacDecoder.Decode(_rt, _dev, _w, codes, codes.Length / Ar.NCb);

    /// <summary>Text + voice → 24 kHz float32 PCM.</summary>
    public float[] Synthesize(string text, ReferenceVoice voice, HiggsOptions? options = null)
        => Decode(GenerateCodes(text, voice, options));

    /// <summary>Same as <see cref="GenerateCodes"/>, but each completed code frame is
    /// passed to <paramref name="onFrame"/> as soon as the AR produces it. Returning false
    /// stops generation early (client gone). See <see cref="Ar.GenerateStreaming"/>.</summary>
    public void GenerateCodesStreaming(string text, ReferenceVoice voice, HiggsOptions? options,
                                       Func<int[], bool> onFrame)
    {
        options ??= new HiggsOptions();
        if (_tok == null) throw new InvalidOperationException("no tokenizer; call SetTokenizer first");
        if (!string.IsNullOrEmpty(options.StyleTag)) text = options.StyleTag + text;

        var prompt = BuildPrompt(_tok, text, voice.RefText, voice.Rows + Ar.NCb - 1);
        Ar.GenerateStreaming(_rt, _dev, _w, prompt, voice.Codes, voice.Rows,
                             options.Temperature, options.Seed, options.MaxSteps, options.TopK,
                             options.GraphCache, onFrame);
    }

    /// <summary>Text + voice → PCM in chunks as it is produced, instead of one array at
    /// the end. Runs the AR and the DAC on this thread in lockstep (the reference server
    /// decodes inside its AR callback too) — nothing here is concurrent.
    ///
    /// <paramref name="onPcm"/> gets ~<see cref="HiggsOptions.StreamStepFrames"/> frames'
    /// worth of 24 kHz float PCM per call. Returning false stops generation (client
    /// disconnected). Generation also stops on its own after roughly four seconds of
    /// trailing silence, which the AR sometimes produces without ever emitting EOC.
    ///
    /// Returns how many frames' audio was actually handed over, and whether the silence
    /// rule is what ended it.</summary>
    public (int Frames, bool StoppedForSilence) SynthesizeStreaming(
        string text, ReferenceVoice voice, HiggsOptions? options,
        Func<ReadOnlyMemory<float>, bool> onPcm)
        => SynthesizeStreamingCodes(text, voice, options, null, onPcm);

    /// <summary>As <see cref="SynthesizeStreaming"/>, but also hands every generated code
    /// frame to <paramref name="onFrame"/> — so a caller can keep the codes that produced
    /// the audio it just heard, instead of having to reproduce the run.</summary>
    public (int Frames, bool StoppedForSilence) SynthesizeStreamingCodes(
        string text, ReferenceVoice voice, HiggsOptions? options,
        Func<int[], bool>? onFrame, Func<ReadOnlyMemory<float>, bool> onPcm)
    {
        var dac = NewStream(options, onPcm);
        GenerateCodesStreaming(text, voice, options, frame =>
            (onFrame == null || onFrame(frame)) && dac.Push(frame, onPcm));
        FinishStream(dac, onPcm);
        return (dac.DecodedFrames, dac.StoppedForSilence);
    }

    /// <summary>Decode an already-generated code block in streaming windows rather than
    /// in one pass. Same audio as <see cref="Decode"/> apart from the decode-window
    /// boundaries; exists so a fixed code block can be run down both paths and compared
    /// (<c>decode --codes ... --stream</c>).</summary>
    public (int Frames, bool StoppedForSilence) DecodeStreaming(
        int[] codes, HiggsOptions? options, Func<ReadOnlyMemory<float>, bool> onPcm)
    {
        var dac = NewStream(options, onPcm);
        int n = codes.Length / Ar.NCb;
        for (int t = 0; t < n; t++)
        {
            var frame = new int[Ar.NCb];
            Array.Copy(codes, t * Ar.NCb, frame, 0, Ar.NCb);
            if (!dac.Push(frame, onPcm)) break;
        }
        FinishStream(dac, onPcm);
        return (dac.DecodedFrames, dac.StoppedForSilence);
    }

    private DacStream NewStream(HiggsOptions? options,
                                Func<ReadOnlyMemory<float>, bool> onPcm)
    {
        var dac = new DacStream(_rt, _dev, _w);
        if (options != null)
        {
            dac.Lookahead = options.StreamLookaheadFrames;
            dac.Step = options.StreamStepFrames;
            dac.SilenceStopSamples = options.SilenceStopSamples;
        }
        return dac;
    }

    private static void FinishStream(DacStream dac, Func<ReadOnlyMemory<float>, bool> onPcm)
    {
        // Nothing left to hold silence back for, so a stop here would only drop the tail.
        if (!dac.StoppedForSilence) dac.Flush(onPcm);
    }

    private static int[] BuildPrompt(HiggsTokenizer tok, string text, string? refText, int numRef)
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

    public string? BackendName => _rt.Name;

    /// <summary>Number of weights loaded (introspection).</summary>
    public int TensorCount => _w.Count;

    /// <summary>Tokenize text with the loaded BPE (introspection; throws if none set).</summary>
    public int[] EncodeText(string text)
        => _tok != null ? _tok.Encode(text) : throw new InvalidOperationException("no tokenizer");

    public void Dispose()
    {
        if (_disposed) return;
        _w.Dispose();
        _rt.Dispose();
        _disposed = true;
    }
}
