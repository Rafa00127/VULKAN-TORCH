using System;
using VulkanTorch;

internal static class Program
{
    private static byte[] Bytes(float[] a)
    {
        var b = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        return b;
    }

    private static float[] NaiveMatmul(float[] a, float[] b, int M, int K, int N)
    {
        var c = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double s = 0;
                for (int k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n];
                c[m * N + n] = (float)s;
            }
        return c;
    }

    private static float[] NaiveConv1d(float[] x, float[] w, int T, int Cin, int Cout, int K, int pad)
    {
        int Tout = T + 2 * pad - (K - 1);
        var y = new float[Tout * Cout];
        for (int t = 0; t < Tout; t++)
            for (int oc = 0; oc < Cout; oc++)
            {
                double s = 0;
                for (int k = 0; k < K; k++)
                {
                    int ti = t + k - pad;
                    if (ti < 0 || ti >= T) continue;
                    for (int ic = 0; ic < Cin; ic++)
                        s += x[ti * Cin + ic] * w[(oc * Cin + ic) * K + k];
                }
                y[t * Cout + oc] = (float)s;
            }
        return y;
    }

    private static float MaxErr(float[] refv, float[] got)
    {
        float e = 0;
        for (int i = 0; i < refv.Length; i++) e = Math.Max(e, Math.Abs(refv[i] - got[i]));
        return e;
    }

    private static int Main()
    {
        using var rt = new Runtime();
        var dev = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");

        var rng = new Random(1);
        int fails = 0;

        // ---- matmul ----
        {
            int M = 8, K = 6, N = 5;
            var A = new float[M * K];
            var B = new float[K * N];
            for (int i = 0; i < A.Length; i++) A[i] = (float)rng.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)rng.NextDouble();

            using var wm = new Memory(dev, 1UL << 20);
            var b = wm.Tensor(new long[] { K, N }, Ops.F32, Bytes(B));
            using var g = new Graph(rt, dev);
            g.Enter();
            var a = g.Input(new long[] { M, K }, A);
            var c = Ops.Contiguous(Ops.Matmul(a, b)).MarkOutput();
            var got = c.ToFloats(g);
            g.Exit();

            float e = MaxErr(NaiveMatmul(A, B, M, K, N), got);
            Console.WriteLine($"[{(e < 1e-4 ? " OK " : "FAIL")}] matmul  max_err={e:E2}  buf={c.BackendName}");
            if (e >= 2e-3) fails++;
        }

        // ---- conv1d ----
        {
            int T = 32, Cin = 4, Cout = 6, K = 5, pad = 2;
            var X = new float[T * Cin];
            var W = new float[Cout * Cin * K];
            for (int i = 0; i < X.Length; i++) X[i] = (float)rng.NextDouble();
            for (int i = 0; i < W.Length; i++) W[i] = (float)rng.NextDouble();

            using var wm = new Memory(dev, 1UL << 20);
            var w = wm.Tensor(new long[] { Cout, Cin, K }, Ops.F32, Bytes(W));
            using var g = new Graph(rt, dev);
            g.Enter();
            var x = g.Input(new long[] { T, Cin }, X);
            var y = Ops.Contiguous(Ops.Conv1d(x, w, 1, pad, 1)).MarkOutput();
            var got = y.ToFloats(g);
            g.Exit();

            float e = MaxErr(NaiveConv1d(X, W, T, Cin, Cout, K, pad), got);
            Console.WriteLine($"[{(e < 2e-3 ? " OK " : "FAIL")}] conv1d  max_err={e:E2}");
            if (e >= 2e-3) fails++;
        }

        Console.WriteLine(fails == 0 ? "\nall passed" : "\nsome FAILED");
        return fails == 0 ? 0 : 1;
    }
}
