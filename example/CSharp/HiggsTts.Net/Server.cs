using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using HiggsTts;
using HiggsTtsSharp;

namespace HiggsTts;

/// <summary>
/// OpenAI-compatible speech endpoint over the <see cref="Tts"/> facade:
///
///   POST /v1/audio/speech   {"model","input","voice","response_format","speed"}
///
/// Audio is produced and streamed as it is generated — the response body starts
/// arriving roughly a second in, not at the end. <c>response_format: "pcm"</c> (the
/// default) gives raw PCM the way OpenAI does it: signed 16-bit little-endian, mono,
/// 24 kHz, no container. <c>"wav"</c> prepends a RIFF header.
///
/// Everything runs on one request at a time: the engine serializes on a shared
/// scheduler, and <see cref="Tts"/> is documented not thread-safe. Requests queue.
/// </summary>
internal static class Server
{
    private const int SampleRate = Tts.SampleRate;

    private sealed class Voice
    {
        public required string Wav;
        public string? RefText;
        public ReferenceVoice Encoded = null!;   // filled by Warmup, before the port opens
    }

    public static int Run(string model, string? tokenizer, string defaultWav, string? defaultRefText,
                          IEnumerable<string> voiceSpecs, int port)
    {
        Console.OutputEncoding = Encoding.UTF8;   // logs go to a file redirected as cp936 otherwise

        // The reference voice is encoded once and reused for every request; encoding is
        // the expensive part of a request and is text-independent.
        var voices = new Dictionary<string, Voice>(StringComparer.Ordinal);
        voices["default"] = new Voice { Wav = defaultWav, RefText = defaultRefText };
        foreach (var spec in voiceSpecs)
        {
            int eq = spec.IndexOf('=');
            if (eq <= 0 || eq == spec.Length - 1)
            {
                Console.Error.WriteLine($"serve: --voice expects name=<wav>[,<transcript>], got '{spec}'");
                return 2;
            }
            var rest = spec[(eq + 1)..];
            int comma = rest.IndexOf(',');
            voices[spec[..eq]] = new Voice
            {
                Wav = comma < 0 ? rest : rest[..comma],
                RefText = comma < 0 ? null : rest[(comma + 1)..],
            };
        }

        using var tts = new Tts(model, tokenizer);
        Console.WriteLine($"backend: {tts.BackendName}");

        // A voice's RVQ codes depend only on the reference audio, never on the text being
        // synthesized, so encode every configured voice now — before the port opens.
        // Otherwise whichever request arrives first silently pays ~125 ms extra, which is
        // exactly the request whose latency anyone testing this will notice.
        foreach (var (name, voice) in voices)
        {
            var swv = Stopwatch.StartNew();
            voice.Encoded = tts.EncodeReference(Cli.LoadWav(voice.Wav, out int sr), sr, voice.RefText);
            Console.WriteLine($"voice '{name}': {voice.Encoded.Rows} frames "
                              + $"({swv.Elapsed.TotalMilliseconds:F0} ms)  {voice.Wav}");
        }

        var listener = new HttpListener();
        string prefix = $"http://127.0.0.1:{port}/";
        listener.Prefixes.Add(prefix);
        try
        {
            listener.Start();
        }
        catch (HttpListenerException e)
        {
            Console.Error.WriteLine(
                $"serve: cannot listen on {prefix}: {e.Message}\n" +
                $"       If this is the http.sys ACL, either run once as administrator:\n" +
                $"         netsh http add urlacl url={prefix} user=%USERNAME%\n" +
                $"       or pick another port with --port.");
            return 1;
        }
        Console.WriteLine($"listening on {prefix}  (POST /v1/audio/speech)");

        while (true)
        {
            HttpListenerContext ctx;
            try
            {
                ctx = listener.GetContext();
            }
            catch (HttpListenerException)
            {
                break;   // listener stopped
            }
            try
            {
                Handle(tts, voices, ctx);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"serve: request failed: {e.Message}");
                try { ctx.Response.StatusCode = 500; ctx.Response.Close(); } catch { /* gone */ }
            }
        }
        return 0;
    }

    private static void Handle(Tts tts, Dictionary<string, Voice> voices, HttpListenerContext ctx)
    {
        var req = ctx.Request;
        var res = ctx.Response;
        if (!string.Equals(req.HttpMethod, "POST", StringComparison.OrdinalIgnoreCase) ||
            !req.Url!.AbsolutePath.TrimEnd('/').Equals("/v1/audio/speech", StringComparison.OrdinalIgnoreCase))
        {
            res.StatusCode = 404;
            Json(res, """{"error":{"message":"POST /v1/audio/speech only"}}""");
            return;
        }

        string body;
        using (var r = new StreamReader(req.InputStream, Encoding.UTF8))
            body = r.ReadToEnd();

        string? input, voiceName = null, format = null;
        double speed = 1.0;
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            input = Str(root, "input");
            voiceName = Str(root, "voice");
            format = Str(root, "response_format");
            var sp = root.TryGetProperty("speed", out var spEl) && spEl.ValueKind == JsonValueKind.Number
                ? spEl.GetDouble() : 1.0;
            speed = sp;
        }
        catch (JsonException e)
        {
            Bad(res, $"request body is not JSON: {e.Message}");
            return;
        }

        if (string.IsNullOrEmpty(input)) { Bad(res, "'input' is required"); return; }
        if (Math.Abs(speed - 1.0) > 1e-6)
        {
            // Time-stretching would change the code stream, not just the sample rate.
            Bad(res, "'speed' is not supported (only 1.0)");
            return;
        }

        format ??= "pcm";
        if (format is not ("pcm" or "wav"))
        {
            Bad(res, $"'response_format' must be 'pcm' or 'wav', got '{format}'");
            return;
        }
        bool wav = format == "wav";

        if (!voices.TryGetValue(voiceName ?? "default", out var voice))
        {
            Bad(res, $"unknown voice '{voiceName}' (have: {string.Join(", ", voices.Keys)})");
            return;
        }

        res.StatusCode = 200;
        res.ContentType = wav ? "audio/wav" : "audio/pcm";
        res.SendChunked = true;
        var outStream = res.OutputStream;

        var sw = Stopwatch.StartNew();
        long firstMs = -1, samples = 0;
        bool clientGone = false;
        var pcm = new byte[0];

        try
        {
            var refVoice = voice.Encoded;

            // A RIFF header has to state the payload length up front, but the length is not
            // known until generation ends, and HttpListener's response stream cannot seek
            // back to patch it. So "wav" buffers the audio and sends it in one piece; "pcm"
            // needs no lengths and streams chunk by chunk as intended.
            var buffer = wav ? new MemoryStream() : null;

            // One synthesizer at a time: the engine holds a single scheduler.
            lock (Lock)
            {
                int chunks = 0, arFrames = 0;
                long lastMs = 0, worstMs = 0;
                var allCodes = new List<int>();
                var (frames, stopped) = tts.SynthesizeStreamingCodes(input, refVoice, null,
                    f => { arFrames++; allCodes.AddRange(f); return true; },
                    chunk =>
                {
                    long now = (long)sw.Elapsed.TotalMilliseconds;
                    if (firstMs < 0) firstMs = now;
                    chunks++;
                    long gap = now - lastMs;
                    if (gap > worstMs) worstMs = gap;
                    lastMs = now;
                    var src = chunk.Span;
                    if (pcm.Length < src.Length * 2) pcm = new byte[src.Length * 2];
                    for (int i = 0; i < src.Length; i++)
                    {
                        // No tanh in this decoder, so |x| can exceed 1; clamp or the
                        // cast wraps and every peak becomes a click.
                        short v = (short)Math.Round(Math.Clamp(src[i], -1f, 1f) * 32767f);
                        BinaryPrimitives.WriteInt16LittleEndian(pcm.AsSpan(2 * i), v);
                    }
                    samples += src.Length;
                    var pcmChunk = pcm.AsSpan(0, src.Length * 2);
                    if (buffer != null) { buffer.Write(pcmChunk); return true; }
                    clientGone = !WriteAll(outStream, clientGone, pcmChunk);
                    return !clientGone;
                });
                Console.WriteLine($"serve: '{Trim(input)}' voice={voiceName ?? "default"} "
                                  + $"-> {frames} frames, {samples / (double)SampleRate:F2} s, "
                                  + $"first audio {firstMs:F0} ms, total {sw.Elapsed.TotalMilliseconds:F0} ms"
                                  + (stopped ? ", stopped on trailing silence" : ""));
                if (Environment.GetEnvironmentVariable("HIGGS_DEBUG") == "1")
                    Console.WriteLine($"       ar-frames={arFrames} decoded-frames={frames} "
                                      + $"chunks={chunks} worst-gap={worstMs} ms "
                                      + $"prefill={Ar.LastPrefillMs:F0} ms");
                if (Environment.GetEnvironmentVariable("HIGGS_DUMP_CODES") is { } dumpPath)
                {
                    var cb = new byte[allCodes.Count * 4];
                    Buffer.BlockCopy(allCodes.ToArray(), 0, cb, 0, cb.Length);
                    File.WriteAllBytes(dumpPath, cb);
                    Console.WriteLine($"       dumped {allCodes.Count / 8} code frames -> {dumpPath}");
                }
            }
            if (buffer != null && !clientGone)
            {
                WriteAll(outStream, clientGone, WavHeader((uint)(samples * 2)));
                buffer.Position = 0;
                buffer.CopyTo(outStream);
            }
        }
        catch (Exception e)
        {
            // The client vanishing mid-stream is the one failure that is not an error: it
            // is how generation gets stopped early. Anything else is real (a native op
            // failure surfaces here as InvalidOperationException).
            if (clientGone) Console.Error.WriteLine("serve: client disconnected, generation stopped");
            else
            {
                Console.Error.WriteLine($"serve: synthesis failed after {samples} samples: "
                                        + $"{e.GetType().Name}: {e.Message}");
                throw;
            }
        }
        finally
        {
            try { res.Close(); } catch { /* client gone */ }
        }
    }

    private static readonly object Lock = new();

    /// <summary>Encode a voice, timing it (-1 when it was already encoded).</summary>
    private static ReferenceVoice Encode(Tts tts, Voice voice, out long ms)
    {
        var sw = Stopwatch.StartNew();
        var v = tts.EncodeReference(Cli.LoadWav(voice.Wav, out int sr), sr, voice.RefText);
        ms = (long)sw.Elapsed.TotalMilliseconds;
        return v;
    }

    /// <summary>Write, absorbing a disconnect into the return value so the caller can stop
    /// generating instead of throwing out of the AR loop. Returns false once the client is
    /// gone (and never writes again). http.sys surfaces a dropped connection as more than
    /// one exception type depending on where the write failed.</summary>
    private static bool WriteAll(Stream s, bool gone, ReadOnlySpan<byte> data)
    {
        if (gone || data.Length == 0) return !gone;
        try
        {
            s.Write(data);
            return true;
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException
                                      or System.Net.Sockets.SocketException
                                      or InvalidOperationException or HttpListenerException)
        {
            return false;
        }
    }

    /// <summary>Minimal 16-bit PCM mono RIFF/WAVE header for <paramref name="dataBytes"/>
    /// of payload.</summary>
    private static byte[] WavHeader(uint dataBytes)
    {
        var h = new byte[44];
        Encoding.ASCII.GetBytes("RIFF").CopyTo(h, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(4), 36 + dataBytes);   // file size - 8
        Encoding.ASCII.GetBytes("WAVEfmt ").CopyTo(h, 8);
        BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(16), 16);           // fmt chunk size
        BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(20), 1);            // PCM
        BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(22), 1);            // mono
        BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(24), SampleRate);
        BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(28), SampleRate * 2);   // byte rate
        BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(32), 2);            // block align
        BinaryPrimitives.WriteUInt16LittleEndian(h.AsSpan(34), 16);           // bits
        Encoding.ASCII.GetBytes("data").CopyTo(h, 36);
        BinaryPrimitives.WriteUInt32LittleEndian(h.AsSpan(40), dataBytes);
        return h;
    }

    private static string? Str(JsonElement root, string name)
    {
        // Both naming conventions show up: OpenAI/pydantic clients send snake_case,
        // .NET-idiomatic ones send the property name. Try name, then Name.
        foreach (var key in new[] { name, char.ToUpperInvariant(name[0]) + name[1..] })
            if (root.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.String)
                return el.GetString();
        return null;
    }

    private static void Bad(HttpListenerResponse res, string message)
    {
        res.StatusCode = 400;
        Json(res, JsonSerializer.Serialize(new { error = new { message } }));
    }

    private static void Json(HttpListenerResponse res, string body)
    {
        var bytes = Encoding.UTF8.GetBytes(body);
        res.ContentType = "application/json";
        res.ContentLength64 = bytes.Length;
        res.OutputStream.Write(bytes);
        res.Close();
    }

    private static string Trim(string s) => s.Length <= 40 ? s : s[..40] + "...";
}
