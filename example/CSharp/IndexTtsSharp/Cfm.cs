using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 s2mel conditional flow matching: a fixed 25-step Euler ODE solver over the DiT
/// estimator with classifier-free guidance.
/// Ported from indextts/s2mel/modules/flow_matching.py (CFM.solve_euler).
///
/// The estimator graph (both CFG passes) is captured once and replayed for every Euler step:
/// the latent and the timestep embedding are graph inputs, everything else is fixed, so a
/// single <c>AllocStatic</c>/<c>ComputeStatic</c> loop replaces 25 fresh graph builds. The
/// latent is carried on the host between steps. CFG's stacked [x, x] batch is done as two
/// estimator calls (cond / all-zero null), which is numerically identical.
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

        // Capture the estimator graph ONCE (both CFG passes), with the latent and the timestep
        // embedding as graph inputs, then replay it for every Euler step. The reference rebuilds
        // the whole 2-pass graph per step; here only the inputs change between steps.
        using var g = new Graph(_rt, _dev);
        g.Enter();
        var pos = g.InputI32(new long[] { t }, Range(t));
        var xT = g.Input(new long[] { t, Mel }, x);
        var pX = g.Input(new long[] { t, Mel }, promptX);
        var mT = g.Input(new long[] { t, CondDim }, mu);
        var sT = g.Input(new long[] { 1, Style }, style);
        var zX = g.Input(new long[] { t, Mel }, zerosX);
        var zM = g.Input(new long[] { t, CondDim }, zerosMu);
        var zS = g.Input(new long[] { 1, Style }, zerosStyle);
        var fe = g.Input(new long[] { 1, 256 }, Dit.TimeEmbedFreq(0f));

        var dc = _dit.Forward(g, xT, pX, mT, sT, fe, pos);
        var du = _dit.Forward(g, xT, zX, zM, zS, fe, pos);
        var dphiT = Ops.Sub(Ops.Scale(dc, 1f + cfgRate), Ops.Scale(du, cfgRate)).MarkOutput();
        g.Exit();
        g.AllocStatic();

        for (int s = 1; s <= steps; s++)
        {
            float ti = (s - 1) * dt;
            g.SetInput(xT, x);
            g.SetInput(fe, Dit.TimeEmbedFreq(ti));
            g.ComputeStatic();
            var dphi = dphiT.ToFloats(g);
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
