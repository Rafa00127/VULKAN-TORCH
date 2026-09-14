using System;

namespace IndexTts;

/// <summary>
/// Host-side DSP front-end for the reference-audio chain: FFT, windows, mel filter banks, the
/// Kaldi-compatible filterbank (torchaudio.compliance.kaldi.fbank) and the librosa-style mel
/// spectrogram used for the reference mel.
///
/// Everything is deterministic and computed on the host — like the reference, which uses
/// torchaudio/librosa for exactly these two steps.
/// </summary>
internal static class Dsp
{
    /// <summary>torch finfo(float32).eps — the log floor used by both Kaldi fbank and the
    /// SeamlessM4T feature extractor.</summary>
    public const double FloatEps = 1.1920928955078125e-07;

    // ---- FFT ------------------------------------------------------------------------------

    /// <summary>In-place iterative radix-2 Cooley-Tukey FFT. length must be a power of two.</summary>
    public static void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        if ((n & (n - 1)) != 0) throw new ArgumentException("FFT length must be a power of two");
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = -2.0 * Math.PI / len;
            double wr = Math.Cos(ang), wi = Math.Sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double cr = 1.0, ci = 0.0;
                for (int k = 0; k < len / 2; k++)
                {
                    int a = i + k, b = i + k + len / 2;
                    double xr = re[b] * cr - im[b] * ci;
                    double xi = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - xr; im[b] = im[a] - xi;
                    re[a] += xr; im[a] += xi;
                    double ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
            }
        }
    }

    // ---- windows --------------------------------------------------------------------------

    /// <summary>Povey window (Kaldi / HF): hanning^0.85, symmetric.</summary>
    public static double[] Povey(int n)
    {
        var w = new double[n];
        for (int i = 0; i < n; i++)
        {
            double hann = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / (n - 1));
            w[i] = Math.Pow(hann, 0.85);
        }
        return w;
    }

    /// <summary>Hann window (torch.hann_window, periodic).</summary>
    public static double[] Hann(int n)
    {
        var w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / n);
        return w;
    }

    // ---- mel scales -----------------------------------------------------------------------

    /// <summary>Kaldi mel: 1127 * ln(1 + f/700).</summary>
    public static double HzToMelKaldi(double hz) => 1127.0 * Math.Log(1.0 + hz / 700.0);
    public static double MelToHzKaldi(double mel) => 700.0 * (Math.Exp(mel / 1127.0) - 1.0);

    /// <summary>Slaney mel (librosa default): linear below 1 kHz, logarithmic above.</summary>
    public static double HzToMelSlaney(double hz)
    {
        const double fMin = 0.0, fSp = 200.0 / 3.0;
        double minLogHz = 1000.0, minLogMel = (minLogHz - fMin) / fSp;
        double logStep = Math.Log(6.4) / 27.0;
        return hz < minLogHz ? (hz - fMin) / fSp : minLogMel + Math.Log(hz / minLogHz) / logStep;
    }

    public static double MelToHzSlaney(double mel)
    {
        const double fMin = 0.0, fSp = 200.0 / 3.0;
        double minLogHz = 1000.0, minLogMel = (minLogHz - fMin) / fSp;
        double logStep = Math.Log(6.4) / 27.0;
        return mel < minLogMel ? fMin + fSp * mel : minLogHz * Math.Exp(logStep * (mel - minLogMel));
    }

    /// <summary>Triangular filter bank. <paramref name="inMelSpace"/> builds the triangles on a
    /// mel grid (Kaldi / HF SeamlessM4T); otherwise on a linear frequency grid (librosa).
    /// <paramref name="slaneyNorm"/> applies librosa's constant-energy scaling.
    /// Result is [nFilters, nFreqBins].</summary>
    public static double[,] MelFilterBank(int nFreqBins, int nFilters, double fMin, double fMax,
                                          int sr, bool inMelSpace, bool slaneyNorm)
    {
        var f = new double[nFilters + 2];
        if (inMelSpace)
        {
            double mMin = HzToMelKaldi(fMin), mMax = HzToMelKaldi(fMax);
            for (int i = 0; i < f.Length; i++)
                f[i] = MelToHzKaldi(mMin + (mMax - mMin) * i / (nFilters + 1));
        }
        else
        {
            double mMin = HzToMelSlaney(fMin), mMax = HzToMelSlaney(fMax);
            for (int i = 0; i < f.Length; i++)
                f[i] = MelToHzSlaney(mMin + (mMax - mMin) * i / (nFilters + 1));
        }

        var bins = new double[nFreqBins];
        if (inMelSpace)
        {
            double binWidth = (double)sr / ((nFreqBins - 1) * 2);
            for (int i = 0; i < nFreqBins; i++) bins[i] = HzToMelKaldi(binWidth * i);
        }
        else
        {
            for (int i = 0; i < nFreqBins; i++) bins[i] = fMin + (fMax - fMin) * i / (nFreqBins - 1);
        }

        var m = new double[nFilters, nFreqBins];
        for (int k = 0; k < nFilters; k++)
        {
            double lo = inMelSpace ? HzToMelKaldi(f[k]) : f[k];
            double ce = inMelSpace ? HzToMelKaldi(f[k + 1]) : f[k + 1];
            double hi = inMelSpace ? HzToMelKaldi(f[k + 2]) : f[k + 2];
            for (int i = 0; i < nFreqBins; i++)
            {
                double x = bins[i];
                double up = (x - lo) / (ce - lo);
                double down = (hi - x) / (hi - ce);
                m[k, i] = Math.Max(0.0, Math.Min(up, down));
            }
        }
        if (slaneyNorm)
        {
            for (int k = 0; k < nFilters; k++)
            {
                double enorm = 2.0 / (f[k + 2] - f[k]);
                for (int i = 0; i < nFreqBins; i++) m[k, i] *= enorm;
            }
        }
        return m;
    }

    // ---- Kaldi-compatible fbank -----------------------------------------------------------

    /// <summary>Kaldi filterbank, matching torchaudio.compliance.kaldi.fbank defaults
    /// (povey window, preemphasis 0.97, DC removal, power spectrum, log floor = float eps).
    /// Returns [frames, nBins].</summary>
    public static float[,] KaldiFbank(double[] wav, int sr, int nMels,
                                      int frameLength = 400, int frameShift = 160,
                                      double lowFreq = 20.0, double preemph = 0.97)
    {
        const int fftLen = 512;                 // round_to_power_of_two
        // Kaldi builds num_fft_bins = fftLen/2 filters then pads one zero column.
        // 257 = fftLen/2 + 1 bins so the mel bin width is sr/fftLen (Kaldi's convention);
        // the last bin's filters are all zero anyway.
        var banks = MelFilterBank(fftLen / 2 + 1, nMels, lowFreq, sr / 2.0, sr,
                                  inMelSpace: true, slaneyNorm: false);

        int frames = 1 + (wav.Length - frameLength) / frameShift;
        var outp = new float[frames, nMels];
        var win = Povey(frameLength);
        var re = new double[fftLen];
        var im = new double[fftLen];

        for (int t = 0; t < frames; t++)
        {
            int start = t * frameShift;
            double mean = 0;
            for (int i = 0; i < frameLength; i++) mean += wav[start + i];
            mean /= frameLength;

            var buf = new double[frameLength];
            for (int i = 0; i < frameLength; i++) buf[i] = wav[start + i] - mean;
            // preemphasis: buf[i] -= c*buf[i-1], buf[0] *= 1-c
            for (int i = frameLength - 1; i >= 1; i--) buf[i] -= preemph * buf[i - 1];
            buf[0] *= 1.0 - preemph;
            for (int i = 0; i < frameLength; i++) buf[i] *= win[i];

            Array.Clear(re); Array.Clear(im);
            Array.Copy(buf, re, frameLength);
            Fft(re, im);

            for (int k = 0; k < nMels; k++)
            {
                double acc = 0;
                for (int b = 0; b < fftLen / 2 + 1; b++)
                {
                    double p = re[b] * re[b] + im[b] * im[b];
                    acc += banks[k, b] * p;
                }
                outp[t, k] = (float)Math.Log(Math.Max(acc, FloatEps));
            }
        }
        return outp;
    }

    /// <summary>SeamlessM4T feature extractor: 16-bit-scaled Kaldi fbank, per-mel-bin
    /// zero-mean/unit-var normalization, then stride-2 channel stacking. [frames, 160].</summary>
    public static float[,] SeamlessFeatures(double[] wav16k, int sr = 16000)
        => SeamlessFeatures(wav16k, out _, sr);

    /// <summary>As above, also returning the attention mask. The feature extractor pads to a
    /// multiple of the stride, so when the frame count is odd the trailing stacked frame has no
    /// real audio behind it and its mask entry is 0.</summary>
    public static float[,] SeamlessFeatures(double[] wav16k, out float[] mask, int sr = 16000)
    {
        var scaled = new double[wav16k.Length];
        for (int i = 0; i < wav16k.Length; i++) scaled[i] = wav16k[i] * 32768.0;
        var fb = KaldiFbank(scaled, sr, 80);
        return Stack(fb, out mask);
    }

    /// <summary>Per-bin normalise a Kaldi fbank, pad to a multiple of 2 with padding_value 1.0,
    /// then stack adjacent frames: [T, 80] -> ([T/2, 160], mask).</summary>
    public static float[,] Stack(float[,] fbIn, out float[] mask)
    {
        int frames = fbIn.GetLength(0), bins = fbIn.GetLength(1);
        var fb = fbIn;
        NormalizePerBin(fb, frames, bins);

        int padded = frames + (frames % 2);
        int outFrames = padded / 2;
        var outp = new float[outFrames, bins * 2];
        mask = new float[outFrames];
        for (int t = 0; t < outFrames; t++)
        {
            mask[t] = 1f;
            for (int b = 0; b < bins; b++)
            {
                outp[t, b] = fb[2 * t, b];
                outp[t, bins + b] = 2 * t + 1 < frames ? fb[2 * t + 1, b] : 1.0f;
            }
        }
        if (frames % 2 == 1) mask[outFrames - 1] = 0f;
        return outp;
    }

    /// <summary>Reference mel: reflect-pad + hann STFT + librosa mel (Slaney scale/norm) +
    /// log(clamp(x, 1e-5)). Mirrors mel_spectrogram(n_fft=1024, hop=256, win=1024, sr=22050).
    /// Input wav22k; returns [80, frames].</summary>
    public static float[,] MelSpectrogram(double[] wav22k, int nFft = 1024, int nMels = 80,
                                          int sr = 22050, int hop = 256, double fmin = 0.0)
    {
        double fmax = sr / 2.0;
        int pad = (nFft - hop) / 2;                       // 384
        int padded = wav22k.Length + 2 * pad;
        var y = new double[padded];
        for (int i = 0; i < pad; i++) y[i] = wav22k[pad - i];                       // reflect
        Array.Copy(wav22k, 0, y, pad, wav22k.Length);
        for (int i = 0; i < pad; i++) y[padded - 1 - i] = wav22k[wav22k.Length - 2 - i];

        int frames = 1 + (padded - nFft) / hop;
        var basis = MelFilterBank(nFft / 2 + 1, nMels, fmin, fmax, sr,
                                  inMelSpace: false, slaneyNorm: true);             // librosa
        var win = Hann(nFft);                                                       // periodic
        var outp = new float[nMels, frames];
        var re = new double[nFft];
        var im = new double[nFft];

        for (int t = 0; t < frames; t++)
        {
            int start = t * hop;
            Array.Clear(re); Array.Clear(im);
            for (int i = 0; i < nFft; i++) re[i] = y[start + i] * win[i];
            Fft(re, im);
            for (int k = 0; k < nMels; k++)
            {
                double acc = 0;
                for (int b = 0; b < nFft / 2 + 1; b++)
                    acc += basis[k, b] * Math.Sqrt(re[b] * re[b] + im[b] * im[b] + 1e-9);
                outp[k, t] = (float)Math.Log(Math.Max(acc, 1e-5));
            }
        }
        return outp;
    }

    /// <summary>Per-mel-bin zero-mean/unit-variance over time, exactly as the HF feature
    /// extractor does for SeamlessM4T: sample variance (ddof=1) with +1e-7 inside the sqrt.</summary>
    public static void NormalizePerBin(float[,] fb, int frames, int bins)
    {
        for (int b = 0; b < bins; b++)
        {
            double mean = 0;
            for (int t = 0; t < frames; t++) mean += fb[t, b];
            mean /= frames;
            double var = 0;
            for (int t = 0; t < frames; t++) { double d = fb[t, b] - mean; var += d * d; }
            var /= frames - 1;
            double sd = Math.Sqrt(var + 1e-7);
            for (int t = 0; t < frames; t++) fb[t, b] = (float)((fb[t, b] - mean) / sd);
        }
    }

    /// <summary>Subtract the per-column (per-bin) mean over time — the CAMPPlus preprocessing
    /// (`feat - feat.mean(dim=0)`).</summary>
    public static void SubtractColumnMean(float[,] fb, int frames, int bins)
    {
        for (int b = 0; b < bins; b++)
        {
            double mean = 0;
            for (int t = 0; t < frames; t++) mean += fb[t, b];
            mean /= frames;
            for (int t = 0; t < frames; t++) fb[t, b] -= (float)mean;
        }
    }
}
