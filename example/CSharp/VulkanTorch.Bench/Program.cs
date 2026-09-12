using System;
using System.Diagnostics;
using VulkanTorch;

// vulkantorch benchmark (C#) — mirrors bench/bench_cpp.cpp and bench/bench_py.py.
// Weights live in GPU Memory; each rep builds one graph, runs the op, and reads
// back only a tiny reduction (sum_rows), so the timing reflects compute rather
// than host<->device transfer. Best of N.
internal static class Program
{
    private const int Reps = 10;

    private static float[] Rand(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)Gauss();
        return a;
    }

    // deterministic standard normal via Box-Muller (matches numpy/mt19937 seed 0 statistically)
    private static double _spare = double.NaN;
    private static readonly Random Rng = new Random(0);
    private static double Gauss()
    {
        if (!double.IsNaN(_spare)) { var s = _spare; _spare = double.NaN; return s; }
        double u, v, sq;
        do { u = 2 * Rng.NextDouble() - 1; v = 2 * Rng.NextDouble() - 1; sq = u * u + v * v; }
        while (sq >= 1 || sq == 0);
        double mul = Math.Sqrt(-2 * Math.Log(sq) / sq);
        _spare = v * mul;
        return u * mul;
    }

    private static byte[] Bytes(float[] a)
    {
        var b = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        return b;
    }

    private static double Matmul(Runtime rt, Device gpu, int m, int k, int n)
    {
        var a = Rand(m * k);
        var b = Rand(k * n);
        using var ma = new Memory(gpu, (ulong)a.Length * 4);
        using var mb = new Memory(gpu, (ulong)b.Length * 4);
        var ta = ma.Tensor(new long[] { m, k }, Ops.F32, Bytes(a));
        var tb = mb.Tensor(new long[] { k, n }, Ops.F32, Bytes(b));

        double best = double.MaxValue;
        for (int r = 0; r < Reps; r++)
        {
            var sw = Stopwatch.StartNew();
            using var g = new Graph(rt, gpu);
            g.Enter();
            var s = Ops.SumRows(Ops.Matmul(ta, tb)).MarkOutput();
            g.Exit();
            s.ToBytes(g);
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static double Conv(Runtime rt, Device gpu, int t, int cin, int cout, int k, int pad)
    {
        var x = Rand(t * cin);
        var w = Rand(cout * cin * k);
        using var mx = new Memory(gpu, (ulong)x.Length * 4);
        using var mw = new Memory(gpu, (ulong)w.Length * 4);
        var tx = mx.Tensor(new long[] { t, cin }, Ops.F32, Bytes(x));
        var tw = mw.Tensor(new long[] { cout, cin, k }, Ops.F32, Bytes(w));

        double best = double.MaxValue;
        for (int r = 0; r < Reps; r++)
        {
            var sw = Stopwatch.StartNew();
            using var g = new Graph(rt, gpu);
            g.Enter();
            var s = Ops.SumRows(Ops.Conv1d(tx, tw, 1, pad, 1)).MarkOutput();
            g.Exit();
            s.ToBytes(g);
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static int Main()
    {
        using var rt = new Runtime();
        var gpu = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");

        Console.WriteLine($"\n{"matmul shape",22} | {"ms",10}");
        Console.WriteLine(new string('-', 38));
        foreach (var (m, k, n) in new[] { (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096) })
            Console.WriteLine($"{$"{m}x{k}x{n}",22} | {Matmul(rt, gpu, m, k, n),7:F2} ms");

        Console.WriteLine($"\n{"conv1d T Cin->Cout",22} | {"ms",10}");
        Console.WriteLine(new string('-', 38));
        foreach (var (t, cin, cout, k, pad) in new[] { (8192, 256, 256, 7, 3), (16384, 512, 512, 7, 3) })
            Console.WriteLine($"{$"{t} {cin}->{cout} K{k}",22} | {Conv(rt, gpu, t, cin, cout, k, pad),7:F2} ms");
        return 0;
    }
}
