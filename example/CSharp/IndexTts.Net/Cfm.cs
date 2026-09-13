using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 s2mel conditional flow matching: a fixed 25-step Euler ODE solver over the DiT
/// estimator with classifier-free guidance.
/// Ported from indextts/s2mel/modules/flow_matching.py (CFM.solve_euler).
///
/// Each Euler step is its own Graph (the full 25-step unrolled graph exceeds ggml's context
/// pool), with the latent carried on the host between steps. CFG's stacked [x, x] batch is done
/// as two estimator calls (cond / all-zero null), which is numerically identical.
/// All host arrays are PT [T, C] row-major.
/// </summary>
public sealed class Cfm
{
    private const int Mel = 80, Style = 192, CondDim = 512;

    private readonly Runtime _rt;
    private readonly Device _dev;
    private readonly Dit _dit;

    public Cfm(Runtime rt, Device dev, Dit dit)
    {
        _rt = rt; _dev = dev; _dit = dit;
    }

    /// <summary>z: PT [T, 80] noise; prompt: PT [Tp, 80]; mu: PT [T, 512]; style: PT [1, 192].
    /// Returns the denoised mel PT [T, 80].</summary>
    public float[] Inference(float[] z, float[] prompt, int promptLen, float[] mu, float[] style,
                             float cfgRate, int steps)
    {
        int t = z.Length / Mel;
        var promptX = new float[t * Mel];
        Array.Copy(prompt, promptX, promptLen * Mel);

        var zerosX = new float[t * Mel];
        var zerosMu = new float[t * CondDim];
        var zerosStyle = new float[Style];

        var x = (float[])z.Clone();
        ZeroPrompt(x, t, promptLen);
        float dt = 1f / steps;

        for (int s = 1; s <= steps; s++)
        {
            float ti = (s - 1) * dt;
            float[] dphi;
            using (var g = new Graph(_rt, _dev))
            {
                g.Enter();
                var pos = g.InputI32(new long[] { t }, Range(t));
                var xT = g.Input(new long[] { t, Mel }, x);
                var pX = g.Input(new long[] { t, Mel }, promptX);
                var mT = g.Input(new long[] { t, CondDim }, mu);
                var sT = g.Input(new long[] { 1, Style }, style);
                var zX = g.Input(new long[] { t, Mel }, zerosX);
                var zM = g.Input(new long[] { t, CondDim }, zerosMu);
                var zS = g.Input(new long[] { 1, Style }, zerosStyle);

                var dc = _dit.Forward(g, xT, pX, mT, sT, ti, pos);
                var du = _dit.Forward(g, xT, zX, zM, zS, ti, pos);
                var dphiT = Ops.Sub(Ops.Scale(dc, 1f + cfgRate), Ops.Scale(du, cfgRate)).MarkOutput();
                dphi = dphiT.ToFloats(g);
                g.Exit();
            }
            for (int i = 0; i < x.Length; i++) x[i] += dt * dphi[i];
            ZeroPrompt(x, t, promptLen);
        }
        return x;
    }

    private static void ZeroPrompt(float[] x, int t, int promptLen)
    {
        for (int i = 0; i < promptLen * Mel && i < x.Length; i++) x[i] = 0f;
    }

    private static int[] Range(int n)
    {
        var v = new int[n];
        for (int i = 0; i < n; i++) v[i] = i;
        return v;
    }
}
