using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace IndexTts;

/// <summary>
/// Minimal reader for numpy ``.npy`` files (the reference tensors dumped by
/// tools/dump_ref_index.py). Handles the formats we emit: C-contiguous
/// float32 / float16 / int32.
/// </summary>
public sealed class Npy
{
    public long[] Shape { get; }
    public float[] Data { get; }

    private Npy(long[] shape, float[] data) { Shape = shape; Data = data; }

    /// <summary>Wrap an in-memory array as an Npy (for reordered reference tensors).</summary>
    public static Npy Wrap(long[] shape, float[] data) => new(shape, data);

    public long Numel
    {
        get { long n = 1; foreach (var s in Shape) n *= s; return n; }
    }

    public static Npy Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);

        var magic = r.ReadBytes(6);
        if (magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY")
            throw new InvalidDataException($"not a .npy file: {path}");
        int major = r.ReadByte();
        r.ReadByte();  // minor
        int hlen = major == 1 ? r.ReadUInt16() : (int)r.ReadUInt32();
        var header = Encoding.ASCII.GetString(r.ReadBytes(hlen));

        string descr = Extract(header, "descr");
        bool fortran = Extract(header, "fortran_order").Contains("True");
        if (fortran) throw new NotSupportedException("fortran-order .npy not supported");

        var shape = new List<long>();
        int ls = header.IndexOf("shape", StringComparison.Ordinal);
        int lp = header.IndexOf('(', ls);
        int rp = header.IndexOf(')', lp);
        foreach (var tok in header.Substring(lp + 1, rp - lp - 1).Split(','))
            if (tok.Trim().Length > 0) shape.Add(long.Parse(tok.Trim()));

        long n = 1;
        foreach (var s in shape) n *= s;
        var data = new float[n];

        switch (descr.Trim('\'', '"'))
        {
            case "<f4":
            case "|f4":
                for (long i = 0; i < n; i++) data[i] = r.ReadSingle();
                break;
            case "<f2":
                for (long i = 0; i < n; i++) data[i] = (float)r.ReadHalf();
                break;
            case "<i4":
            case "|i4":
                for (long i = 0; i < n; i++) data[i] = r.ReadInt32();
                break;
            default:
                throw new NotSupportedException($"npy dtype {descr} not supported");
        }
        return new Npy(shape.ToArray(), data);
    }

    private static string Extract(string header, string key)
    {
        int i = header.IndexOf(key, StringComparison.Ordinal);
        if (i < 0) return "";
        int c = header.IndexOf(':', i);
        int e = header.IndexOf(',', c);
        if (e < 0) e = header.IndexOf('}', c);
        return header.Substring(c + 1, e - c - 1).Trim();
    }
}
