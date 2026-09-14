using System;
using System.IO;
using System.Text;

namespace HiggsTtsSharp;

/// <summary>
/// Serializable reference-voice state for the Higgs engine: the RVQ codes produced by
/// encoding the reference audio, plus the audio's optional transcript. Store it to
/// reuse a voice across many syntheses, or re-encode new audio to change the timbre.
/// </summary>
public sealed class ReferenceVoice
{
    private static readonly byte[] Magic = { (byte)'H', (byte)'G', (byte)'V', 1 };

    /// <summary>Flat RVQ codes, row-major: t0_q0..t0_q7, t1_q0..t1_q7, ...</summary>
    public int[] Codes { get; }

    /// <summary>Transcription of the reference audio (may be null).</summary>
    public string? RefText { get; }

    /// <summary>Number of code frames (each frame has <c>8</c> codes).</summary>
    public int Rows => Codes.Length / 8;

    public ReferenceVoice(int[] codes, string? refText = null)
    {
        if (codes.Length % 8 != 0)
            throw new ArgumentException("codes.Length must be a multiple of 8", nameof(codes));
        Codes = codes;
        RefText = refText;
    }

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.UTF8);
        bw.Write(Magic);
        bw.Write(Codes.Length);
        foreach (var c in Codes) bw.Write(c);
        bw.Write(RefText ?? "");
        bw.Flush();
        return ms.ToArray();
    }

    public static ReferenceVoice FromBytes(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms, Encoding.UTF8);
        var magic = br.ReadBytes(4);
        if (magic.Length != 4 || magic[0] != Magic[0] || magic[1] != Magic[1] ||
            magic[2] != Magic[2] || magic[3] != Magic[3])
            throw new InvalidDataException("not a Higgs reference-voice blob");
        int n = br.ReadInt32();
        var codes = new int[n];
        for (int i = 0; i < n; i++) codes[i] = br.ReadInt32();
        var text = br.ReadString();
        return new ReferenceVoice(codes, text.Length == 0 ? null : text);
    }
}
