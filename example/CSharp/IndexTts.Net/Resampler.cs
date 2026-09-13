using System;

namespace IndexTts;

/// <summary>
/// Rational polyphase resampler (Kaiser-windowed sinc), used for the 24k -> 16k
/// reference-audio path. Self-contained — no libsoxr/NAudio needed. Close enough
/// to librosa's soxr_hq that the resulting RVQ codes agree (the reference audio is
/// a voice timbre hint, not a bit-exact input).
/// </summary>
internal static class Resampler
{
    private const int Taps = 48;          // taps per polyphase branch
    private const double Beta = 8.6;      // Kaiser beta ~ -90 dB stopband

    public static float[] Resample(float[] x, int srIn, int srOut)
    {
        if (srIn == srOut) return x;
        int g = Gcd(srIn, srOut);
        int up = srOut / g, down = srIn / g;

        int protoLen = Taps * up;
        double center = (Taps / 2.0) * up;
        double fc = 0.5 / Math.Max(up, down);      // cutoff, relative to input rate
        var h = new double[protoLen];
        double norm = 0;
        for (int k = 0; k < protoLen; k++)
        {
            double xk = (k - center) / up;         // offset in input samples
            double sinc = Math.Abs(xk) < 1e-9 ? 1.0 : Math.Sin(2 * Math.PI * fc * xk) / (Math.PI * xk);
            double win = Kaiser(k * 2.0 / (protoLen - 1) - 1.0);
            h[k] = sinc * win;
            norm += h[k];
        }
        for (int k = 0; k < protoLen; k++) h[k] /= norm;

        int outLen = (int)((long)x.Length * up / down);
        var y = new float[outLen];
        for (int m = 0; m < outLen; m++)
        {
            long t = (long)m * down;
            long baseIn = t / up;
            int phase = (int)(t % up);
            double acc = 0;
            for (int i = 0; i < Taps; i++)
            {
                long xi = baseIn + i - Taps / 2;
                if (xi < 0 || xi >= x.Length) continue;
                acc += x[xi] * h[i * up + phase];
            }
            y[m] = (float)acc;
        }
        return y;
    }

    private static double Kaiser(double t)
        => I0(Beta * Math.Sqrt(Math.Max(0.0, 1.0 - t * t))) / I0(Beta);

    private static double I0(double x)
    {
        double s = 1, term = 1;
        for (int k = 1; k < 50; k++)
        {
            double r = x / (2.0 * k);
            term *= r * r;
            s += term;
            if (term < 1e-18 * s) break;
        }
        return s;
    }

    private static int Gcd(int a, int b) { while (b != 0) { (a, b) = (b, a % b); } return a; }
}
