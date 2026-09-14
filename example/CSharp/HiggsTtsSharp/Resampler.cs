using System;
using System.Collections.Generic;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace HiggsTts;

/// <summary>
/// Mono float[] resampler backed by NAudio's WDL polyphase resampler
/// (<see cref="WdlResamplingSampleProvider"/>). Used for the reference-audio path
/// (e.g. 44.1k/24k -> 24k/16k). Replaces the earlier hand-rolled Kaiser resampler,
/// which had a wrong cutoff + gain and mis-registered polyphase taps, producing
/// near-silent output at 44.1k->24k and audible distortion elsewhere.
/// </summary>
internal static class Resampler
{
    public static float[] Resample(float[] x, int srIn, int srOut)
    {
        if (srIn == srOut) return x;

        var src = new FloatArrayProvider(x, srIn);
        var dst = new WdlResamplingSampleProvider(src, srOut);

        var outList = new List<float>((int)((long)x.Length * srOut / srIn) + 8);
        var buf = new float[4096];
        int n;
        while ((n = dst.Read(buf, 0, buf.Length)) > 0)
            for (int i = 0; i < n; i++) outList.Add(buf[i]);
        return outList.ToArray();
    }

    /// <summary>A one-shot mono float stream over a fixed array (NAudio source end).</summary>
    private sealed class FloatArrayProvider : ISampleProvider
    {
        private readonly float[] _data;
        private int _pos;

        public FloatArrayProvider(float[] data, int sampleRate)
        {
            _data = data;
            WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        }

        public WaveFormat WaveFormat { get; }

        public int Read(float[] buffer, int offset, int count)
        {
            int n = Math.Min(count, _data.Length - _pos);
            if (n <= 0) return 0;
            Array.Copy(_data, _pos, buffer, offset, n);
            _pos += n;
            return n;
        }
    }
}
