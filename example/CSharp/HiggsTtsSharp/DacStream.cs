using System;
using System.Collections.Generic;
using VulkanTorch;

namespace HiggsTts;

/// <summary>
/// Streaming RVQ-codes → PCM. Frames arrive one at a time (the AR emits frame <c>t</c>
/// N_CB steps after it starts), and audio goes out in chunks as it becomes *stable*.
///
/// A single frame cannot be decoded on its own: the DAC's convolutions are symmetric
/// (non-causal) and its receptive field spans many frames, so a frame's samples depend
/// on both neighbours. The scheme — ported from the reference TCP server
/// (higgs_server.cpp `stream_available_pcm`) — therefore re-decodes a window
/// <c>[decoded - Lookahead, available)</c> on every step and emits only the newly
/// stable <c>[decoded, available - Lookahead)</c>. The overlap is real frames on both
/// sides of what is emitted, so results converge on a one-shot decode as Lookahead
/// grows (see <see cref="DefaultLookahead"/> for the measured numbers).
///
/// The window is deliberately *variable* length: a fixed padded window would put its
/// own edge inside the emitted range. Decode costs ~19 ms for a whole utterance, so
/// rebuilding the graph per chunk is far cheaper than that would be.
/// </summary>
public sealed class DacStream
{
    public const int SamplesPerFrame = 960;   // 25 frames/s at 24 kHz

    /// <summary>Frames of context a decode window carries on each side of what it emits.
    /// Trades first-audio latency against how closely the windowed decode tracks a one-shot
    /// decode of the same codes — see <see cref="ReferenceLookahead"/>.</summary>
    public int Lookahead { get; set; } = DefaultLookahead;

    /// <summary>Lookahead matching the reference C++ server (higgs_server.cpp
    /// kStreamLookaheadFrames), i.e. 640 ms of context and first audio at ~0.96 s.</summary>
    public const int ReferenceLookahead = 16;

    /// <summary>Lookahead this port ships with: 32 frames of context. The DAC's receptive
    /// field reaches much further than its frame rate suggests — every ConvTranspose1d in
    /// the upsampling stack multiplies the backwards reach by its stride — so a window
    /// never reproduces a one-shot decode exactly, it only gets close.
    ///
    /// Measured on the same 99-frame block, relative error vs. the one-shot decode:
    /// 1.1e-2 at 16 (the reference implementation's value), 8.9e-3 at 24, 4.9e-3 at 32,
    /// 4.3e-3 at 48, bit-exact at 64. That error is only worth caring about next to what
    /// the model already gets wrong: a one-shot decode of the official torch reference
    /// codes differs by 8.25e-3 on the f16 GGUF and 8.95e-3 on q8_0 — so below 32 the
    /// windowing is *larger* than the model's own error, and from 32 up it is smaller
    /// (the f16/q8 gap being tiny says the noise is decoder accumulation, not weight
    /// precision). 32 is the cheapest lookahead that hides behind that floor; first
    /// audio lands ~0.67 s in.
    ///
    /// The error is spread evenly over every frame, not concentrated at chunk seams, so
    /// it is not a periodic artefact. <see cref="ReferenceLookahead"/> is as far the other
    /// way as the reference implementation went; 64 buys bit-exactness for 2.5x the wait.</summary>
    public const int DefaultLookahead = 32;

    /// <summary>Frames accumulated before a decode is triggered (one chunk per Step frames).
    /// Each decode costs the same fixed graph build regardless, and the arithmetic is a
    /// few milliseconds, so this is paid for in CPU per chunk, not audio latency —
    /// halving it halves the time to the *second* chunk, not the first.</summary>
    public int Step { get; set; } = 8;

    /// <summary>Stop generating once this many samples of trailing silence have piled up.</summary>
    public int SilenceStopSamples { get; set; } = 4 * 24000;

    /// <summary>Below this per-chunk RMS, a chunk counts as silence.</summary>
    public float SilenceRms { get; set; } = 0.001f;

    private readonly Runtime _rt;
    private readonly Device _dev;
    private readonly HiggsWeights _w;
    private readonly List<int> _codes = new();          // flat, growing: frame f at [f*8, f*8+8)
    private readonly List<float> _pending = new();      // quiet audio held back, see Push
    private int _decoded;                               // frames whose audio has been handed out

    /// <summary>Set when trailing silence crossed <see cref="SilenceStopSamples"/>.</summary>
    public bool StoppedForSilence { get; private set; }

    /// <summary>Frames whose audio has actually been handed to the sink.</summary>
    public int DecodedFrames => _decoded;

    public DacStream(Runtime rt, Device dev, HiggsWeights w)
    {
        _rt = rt; _dev = dev; _w = w;
    }

    /// <summary>Feed one completed code frame (8 codes). Returns false when the sink
    /// asked to stop, or when trailing silence ended the stream — in both cases the
    /// caller should stop generating.</summary>
    public bool Push(int[] frame, Func<ReadOnlyMemory<float>, bool> onPcm)
    {
        _codes.AddRange(frame);
        return Drain(onPcm, final: false);
    }

    /// <summary>Emit whatever is left after the AR loop ends.</summary>
    public bool Flush(Func<ReadOnlyMemory<float>, bool> onPcm) => Drain(onPcm, final: true);

    /// <summary>Decode and emit everything that can be emitted at this point. With
    /// <paramref name="final"/> the lookahead requirement is dropped and the tail goes out.</summary>
    private bool Drain(Func<ReadOnlyMemory<float>, bool> onPcm, bool final)
    {
        int available = _codes.Count / Ar.NCb;

        // Steady state decodes every Step frames, each emitting Step of them, so the
        // window stays Lookahead+Step .. 2*Lookahead+Step frames — bounded and small.
        // At the end there is nothing left to leave lookahead for, so emit it all.
        int stableEnd = final ? available : available - Lookahead;
        if (stableEnd <= _decoded) return true;
        if (!final && stableEnd - _decoded < Step) return true;

        int windowStart = Math.Max(0, _decoded - Lookahead);
        var window = new int[(available - windowStart) * Ar.NCb];
        _codes.CopyTo(windowStart * Ar.NCb, window, 0, window.Length);
        var pcm = DacDecoder.Decode(_rt, _dev, _w, window, available - windowStart);

        int offset = (_decoded - windowStart) * SamplesPerFrame;
        int count = (stableEnd - _decoded) * SamplesPerFrame;
        var emitted = new ReadOnlyMemory<float>(pcm, offset, count);
        _decoded = stableEnd;

        if (IsQuiet(emitted) && !final)
        {
            // Hold quiet audio back rather than sending it: if speech resumes, it is
            // flush()ed first, so nothing is dropped; if it does not, the stream ends
            // early with the silence never having been sent at all.
            _pending.AddRange(emitted.ToArray());
            if (SilenceStopSamples > 0 && _pending.Count >= SilenceStopSamples)
            {
                StoppedForSilence = true;
                return false;
            }
            return true;
        }
        return Emit(onPcm, emitted);
    }

    /// <summary>Send held-back silence (if any), then this chunk.</summary>
    private bool Emit(Func<ReadOnlyMemory<float>, bool> onPcm, ReadOnlyMemory<float> pcm)
    {
        if (_pending.Count > 0)
        {
            if (!onPcm(_pending.ToArray())) return false;
            _pending.Clear();
        }
        return onPcm(pcm);
    }

    private bool IsQuiet(ReadOnlyMemory<float> pcm)
    {
        if (pcm.Length == 0) return true;
        var s = pcm.Span;
        double sum = 0;
        for (int i = 0; i < s.Length; i++) sum += (double)s[i] * s[i];
        return Math.Sqrt(sum / s.Length) < SilenceRms;
    }
}
