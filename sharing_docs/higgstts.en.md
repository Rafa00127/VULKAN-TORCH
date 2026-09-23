# HiggsTTS v3 · Port

**🌏 Language / 语言:** [中文](higgstts.md) · **English**

A port of [HiggsTTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) (a 4B TTS) to
[vulkan-torch](../README.en.md), implemented once per library:

| | Model code | Runnable entry point |
|---|---|---|
| **Python** | [`python/higgstts_vt/`](../example/python/higgstts_vt/): `model.py` / `ar.py` / `decode.py` / `tts.py` | `cli.py` |
| **C#** | [`CSharp/HiggsTtsSharp/`](../example/CSharp/HiggsTtsSharp/): `EncodeRef.cs` / `Ar.cs` / `DacDecoder.cs` / `HiggsTokenizer.cs` / `Resampler.cs` | [`example/CSharp/HiggsTts.Net/`](../example/CSharp/HiggsTts.Net/) (`Cli.cs`) |

Both are self-contained (the C# side ships its own tokenizer and resampler — no Python-side
export needed) and independent of each other.

---

## Usage

Both need **`--model`**, and every other flag has the same name and meaning on both sides (the
full argument table is in the root [README](../README.en.md); `-h` prints it too).

**Python** — [`cli.py`](../example/python/higgstts_vt/cli.py):

```bat
python example\python\higgstts_vt\cli.py ^
  --model path\to\HiggsTTS3-q8_0.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

**C#** — [`HiggsTts.Net`](../example/CSharp/HiggsTts.Net/):

```bat
dotnet build example\CSharp\HiggsTts.Net -c Release

example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe synth ^
  --model D:\models\HiggsTTS3-q8_0.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

`--ref-wav` (defaults to `data/ref_audio/melinaref_24k.wav`), `--ref-text`, `--text` and `--out`
all have defaults, so the minimal call is `--model` plus one `--text`. Where they differ:

| | Python | C# |
|---|---|---|
| `encode_ref` only | `--encode-only` | positional mode `encode` (writes `<out>.i32` of codes instead of synthesizing) |
| default `--out` | `data/higgstts/cli_py.wav` | `data/higgstts/cli_cs_synth.wav` |
| tokenizer | defaults to the in-repo `data/ref_audio/higgs_tts_v3_tokenizer.json` | without `--tokenizer`, built from the GGUF's vocab/merges |

The sampling flags `--temperature` (0.9) / `--topk` (50) / `--seed` (42) / `--max-steps`
(0 = predict from text length) and `--no-graph-cache` are the same on both sides.

**Streaming HTTP endpoint** (C#'s `serve` mode), OpenAI-compatible, audio out as it is generated:

```bat
example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe serve ^
  --model D:\models\HiggsTTS3-q8_0.gguf --port 8000 ^
  --ref-wav data\ref_audio\melinaref_24k.wav --ref-text "I have no doubt you will become Elden Lord." ^
  --voice alice=D:\voices\alice.wav

REM another shell:
curl -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" ^
  -d "{\"input\":\"Hello there.\",\"voice\":\"default\",\"response_format\":\"pcm\"}" -o out.pcm
```

`POST /v1/audio/speech`

`--voice name=<wav>[,<transcript>]` registers an extra voice

---

## Pipeline

```
reference audio ──► encode_ref ──► RVQ codes ──┐(their length: how many audio
                   (codec)                     │ placeholders to reserve)
                                                └──────────┐
                                                           ▼
text / ref text ──► tokenizer ──► ids ──► build_prompt ──► prompt ──┐
                                                                    ├──► Qwen3 36-layer AR ──► codes ──► DAC decode ──► wav
              RVQ codes (the codes themselves, fed alongside prompt)─┘        (backbone)                   (decoder)
```

---

## Weights

[NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) (~4 GB), or convert it yourself:

```bat
python tools\convert\convert-higgs-tts-to-gguf.py --input <HF safetensors dir> --output higgs.gguf
```

**There's a trap in the F16 conversion** (the script already handles it — don't reintroduce it if
you touch it): the 1-D tensors (RMSNorm weights, biases, snake-alpha) and the ten `conv_t` wperm
2-D tensors must stay **F32**. Downcasting them to F16 makes the loader read them expecting F32
byte counts and abort with `GGML_ASSERT(tensor read out of bounds)` — i.e. `--outtype f16` should
quantize 2-D weights only, not "downcast everything".

---

## How the two implementations line up

DAC decode and the AR logits are **bit-identical**, and the tokenizer output matches too; with
the same resampler, `encode_ref` gives 97% identical frames (the only diff comes from Python
using librosa vs C# using its own Kaiser resampler — inaudible).

Speed is about **0.2–0.3 RTF** (on my 7900 xtx): `Decode` is basically free (41 ms — all the time is in the
36-layer backbone AR). For the other two models, see [example-models.en.md](example-models.en.md).
