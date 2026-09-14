using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace VulkanTorch;

/// <summary>
/// Minimal GGUF metadata (header KV block) reader. Reads only the requested keys and
/// skips everything else in place — no tensor data is touched, and no model is loaded.
/// Handy for pulling tokenizer vocab/merges (string arrays) or scalar config straight
/// out of a GGUF.
/// </summary>
public static class GgufMetadata
{
    private const int TypeUInt8 = 0, TypeInt8 = 1, TypeUInt16 = 2, TypeInt16 = 3;
    private const int TypeUInt32 = 4, TypeInt32 = 5, TypeFloat32 = 6, TypeBool = 7;
    private const int TypeString = 8, TypeArray = 9, TypeUInt64 = 10, TypeInt64 = 11, TypeFloat64 = 12;

    /// <summary>
    /// Read the requested keys from the GGUF header. Values are boxed: numeric → long/double,
    /// string → string, bool → bool, array-of-string → string[], other arrays → object[].
    /// </summary>
    public static Dictionary<string, object> Read(string path, params string[] keys)
    {
        var wanted = new HashSet<string>(keys, StringComparer.Ordinal);
        var result = new Dictionary<string, object>(StringComparer.Ordinal);

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8);

        if (br.ReadUInt32() != 0x46554747u)
            throw new InvalidDataException($"Not a GGUF file: {path}");
        uint version = br.ReadUInt32();
        if (version < 2)
            throw new NotSupportedException($"GGUF v{version} is not supported");

        _ = br.ReadUInt64();              // tensor count
        ulong nKv = br.ReadUInt64();      // metadata KV count

        for (ulong i = 0; i < nKv; i++)
        {
            string key = ReadString(br);
            int type = br.ReadInt32();
            if (wanted.Contains(key)) result[key] = ReadValue(br, type);
            else SkipValue(br, type);
        }
        return result;
    }

    /// <summary>Read a string-array metadata key; empty array if absent or not a string array.</summary>
    public static string[] ReadStringArray(string path, string key)
    {
        var meta = Read(path, key);
        return meta.TryGetValue(key, out var v) && v is string[] a ? a : Array.Empty<string>();
    }

    // ── value readers ────────────────────────────────────────────────────

    private static object ReadValue(BinaryReader br, int type) => type switch
    {
        TypeUInt8 => br.ReadByte(),
        TypeInt8 => br.ReadSByte(),
        TypeUInt16 => br.ReadUInt16(),
        TypeInt16 => br.ReadInt16(),
        TypeUInt32 => br.ReadUInt32(),
        TypeInt32 => br.ReadInt32(),
        TypeFloat32 => br.ReadSingle(),
        TypeBool => br.ReadByte() != 0,
        TypeUInt64 => br.ReadUInt64(),
        TypeInt64 => br.ReadInt64(),
        TypeFloat64 => br.ReadDouble(),
        TypeString => ReadString(br),
        TypeArray => ReadArray(br),
        _ => throw new InvalidDataException($"Unknown GGUF value type {type}"),
    };

    private static object ReadArray(BinaryReader br)
    {
        int elemType = br.ReadInt32();
        ulong count = br.ReadUInt64();
        if (elemType == TypeString)
        {
            var s = new string[count];
            for (ulong i = 0; i < count; i++) s[i] = ReadString(br);
            return s;
        }
        var items = new object?[count];
        for (ulong i = 0; i < count; i++) items[i] = ReadValue(br, elemType);
        return items;
    }

    private static string ReadString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        if (len == 0) return "";
        return Encoding.UTF8.GetString(br.ReadBytes((int)len));
    }

    // ── value skippers (no allocation) ───────────────────────────────────

    private static void SkipValue(BinaryReader br, int type)
    {
        switch (type)
        {
            case TypeString: SkipString(br); break;
            case TypeArray: SkipArray(br); break;
            default: br.BaseStream.Seek(FixedSize(type), SeekOrigin.Current); break;
        }
    }

    private static void SkipString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        br.BaseStream.Seek((long)len, SeekOrigin.Current);
    }

    private static void SkipArray(BinaryReader br)
    {
        int elemType = br.ReadInt32();
        ulong count = br.ReadUInt64();
        if (elemType == TypeString)
            for (ulong i = 0; i < count; i++) SkipString(br);
        else if (elemType == TypeArray)
            for (ulong i = 0; i < count; i++) SkipArray(br);
        else
            br.BaseStream.Seek(FixedSize(elemType) * (long)count, SeekOrigin.Current);
    }

    private static long FixedSize(int type) => type switch
    {
        TypeUInt8 or TypeInt8 or TypeBool => 1,
        TypeUInt16 or TypeInt16 => 2,
        TypeUInt32 or TypeInt32 or TypeFloat32 => 4,
        TypeUInt64 or TypeInt64 or TypeFloat64 => 8,
        _ => throw new InvalidDataException($"Not a fixed-size GGUF type: {type}"),
    };
}
