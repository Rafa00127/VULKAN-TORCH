namespace IndexTtsSharp;

/// <summary>Synthesis options (mirror the IndexTTS 2.5 CLI / infer_v2_5.py defaults).</summary>
public sealed class IndexOptions
{
    /// <summary>Text language code: zh (default), en, ja, es, ... (99 languages).</summary>
    public string Lang { get; set; } = "zh";

    /// <summary>Emotion spec, e.g. <c>"happy=0.6,calm=0.4"</c> (null = base emotion from the voice).</summary>
    public string? Emotion { get; set; }

    /// <summary>AR step cap; 0 = <c>max(64, text.Length * 8)</c>.</summary>
    public int MaxSteps { get; set; }

    public int Seed { get; set; } = 42;
    public int TopK { get; set; } = 30;
    public float TopP { get; set; } = 0.8f;
    public float Temperature { get; set; } = 0.8f;
    public float RepPenalty { get; set; } = 10f;

    public float CfgRate { get; set; } = 0.7f;
    public int DiffusionSteps { get; set; } = 25;
    public float DurationFactor { get; set; } = 1.0f;

    /// <summary>Beam-search width (1 = plain sampling).</summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>Replay the captured AR decode graph instead of rebuilding it each step.</summary>
    public bool UseReplay { get; set; } = true;
}
