using System;
using System.IO;
using HiggsTts;
using NAudio.Wave;

internal static class Program
{
    /// <summary>Walk up from the exe to the repo root (marked by CMakeLists.txt + src/).</summary>
    internal static string FindRoot()
    {
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            if (File.Exists(Path.Combine(d.FullName, "CMakeLists.txt")) &&
                Directory.Exists(Path.Combine(d.FullName, "src")))
                return d.FullName;
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("repo root (CMakeLists.txt + src/) not found above the exe");
    }

    internal static void SaveWav(string path, float[] s, int sampleRate = 24000)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var w = new WaveFileWriter(path, new WaveFormat(sampleRate, 16, 1));
        var bytes = new byte[s.Length * 2];
        for (int i = 0; i < s.Length; i++)
        {
            short pcm = (short)Math.Round(Math.Clamp(s[i], -1f, 1f) * 32767f);
            bytes[2 * i] = (byte)pcm;
            bytes[2 * i + 1] = (byte)(pcm >> 8);
        }
        w.Write(bytes, 0, bytes.Length);
    }

    private static int Main(string[] args) => Cli.Run(args);
}
