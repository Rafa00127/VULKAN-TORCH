using System;
using System.IO;
using System.Text;

namespace IndexTtsSharp;

/// <summary>
/// Serializable reference-voice state for the IndexTTS engine: the conditioning tensors
/// extracted from the reference audio. Store it to reuse a voice across syntheses, or
/// re-encode new audio to change the timbre.
/// </summary>
public sealed class ReferenceVoice
{
    private static readonly byte[] Magic = { (byte)'I', (byte)'X', (byte)'V', 1 };

    /// <summary>CAMPPlus timbre embedding, length 192.</summary>
    public float[] Style { get; }

    /// <summary>Wav2Vec2-BERT speaker conditioning, row-major [nFrames, 1024].</summary>
    public float[] Spk { get; }

    /// <summary>22.05 kHz reference mel, row-major [RefFrames, 80].</summary>
    public float[] RefMel { get; }

    /// <summary>Number of reference mel frames (<c>RefMel.Length / 80</c>).</summary>
    public int RefFrames { get; }

    /// <summary>length_regulator(Spk, RefFrames), row-major [RefFrames, 512].</summary>
    public float[] PromptCond { get; }

    public ReferenceVoice(float[] style, float[] spk, float[] refMel, float[] promptCond)
    {
        if (refMel.Length % 80 != 0)
            throw new ArgumentException("refMel.Length must be a multiple of 80", nameof(refMel));
        Style = style;
        Spk = spk;
        RefMel = refMel;
        RefFrames = refMel.Length / 80;
        if (promptCond.Length != RefFrames * 512)
            throw new ArgumentException("promptCond.Length must be RefFrames * 512", nameof(promptCond));
        PromptCond = promptCond;
    }

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.UTF8);
        bw.Write(Magic);
        Write(bw, Style);
        Write(bw, Spk);
        Write(bw, RefMel);
        Write(bw, PromptCond);
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
            throw new InvalidDataException("not an IndexTTS reference-voice blob");
        return new ReferenceVoice(Read(br), Read(br), Read(br), Read(br));
    }

    private static void Write(BinaryWriter bw, float[] a)
    {
        bw.Write(a.Length);
        foreach (var v in a) bw.Write(v);
    }

    private static float[] Read(BinaryReader br)
    {
        int n = br.ReadInt32();
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = br.ReadSingle();
        return a;
    }
}
