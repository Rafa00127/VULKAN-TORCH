# VULKAN-TORCH

**🌏 Language / 语言:** [中文](README.md) · **English**

## Summary & Rambling

Uses part of ggml-vulkan as its core, to build two torch-like libraries (one Python, one C#).

You might be thinking — "wait, this thing is neither ggml nor PyTorch... so it's just plain garbage, huh?"

And honestly? You might be right. But flip it around: it's light (~50 MB) and fast (ggml's got good genes), and lets you quickly port small models **without writing C++** the way raw ggml makes you. That's kind of amazing, no?

Anyway — it's for porting small models for fun. Vulkan backend, so it runs everywhere. Two libraries ship here: `vulkantorch` (Python) and `VulkanTorch.Net` (C#).

They do roughly the same thing; the repo just ports different small models with each.

Yeah yeah, I know — I reinvented the wheel twice (again), to demo porting one TTS model. Should've spent those tokens feeding the fish instead.

The examples port two TTS models and one OCR model (Paddle) — copy them, or just mess around.

Next up, **Little Fat Fish** (小肥鱼) walks you through how to play with these two libraries.

---

## Professor Little Fat Fish's Guide

**vulkan-torch** is a mini tensor library running on ggml's Vulkan backend, with a PyTorch-like API:

- **`vulkantorch`** — Python (pybind11)
- **`VulkanTorch.Net`** — C# (P/Invoke)

Both share the same C++ core (`src/`) and the same ggml, so they behave identically.

| | |
|---|---|
| Purpose | Inference only, no autograd |
| Execution model | eager-style; internally captures each forward pass into one ggml graph, computes on read |
| Devices | explicit `Device` + `Memory` (weights stay resident in VRAM) |
| Layout | PyTorch row-major for the user; bridged internally to ggml's reversed `ne[]` |
| Dependencies | ships ggml (statically linked in); at runtime only a Vulkan driver |

One native per language: `build/vulkantorch.dll` (C#) and `vulkantorch/_vulkantorch*.pyd` (Python) — independent, so a Python-only user doesn't need .NET, and vice versa.

---

## Requirements

| | |
|---|---|
| OS | Any. The C++ core is cross-platform; the repo only ships the Windows build script `build_win.py` — on Linux/mac just use `cmake` |
| Compiler | MSVC / GCC both fine |
| Python | only 3.12 has been tried, but others should work too — the extension is built for whichever interpreter runs `build_win.py` |
| Vulkan | a GPU driver with the Vulkan runtime |
| .NET | SDK 10 (only needed to build the C# examples) |

---

## Build

```bash
python build_win.py
```

Or use GCC (MinGW/w64devkit) via `build.sh` — same thing, just gcc/ninja:

```bash
sh build.sh        # -> build-gcc/, emits build-gcc/vulkantorch.dll
```

Produces:

```
build/vulkantorch.dll                          # shared lib (for C# P/Invoke)
vulkantorch/_vulkantorch.cp312-win_amd64.pyd   # Python extension (built straight into the package)
```

Without pybind11 the Python extension is skipped; `vulkantorch.dll` still builds.

> `build.sh` uses GCC and disables the Python module by default (`BUILD_PYTHON=OFF`) — it only
> emits `vulkantorch.dll`. To build the Python extension too, add `-DBUILD_PYTHON=ON -Dpybind11_DIR=...`.

---

## Directory Layout

```
src/                          C++ core
vulkantorch/                  Python library
VulkanTorch.Net/              C# library
example/                      small-model examples: 2 TTS in C#, 1 TTS + 1 OCR in Python
data/ref_audio/               reference audio
bench/                        some test scripts
third_party/ggml/             ggml, trimmed down to CPU + Vulkan only
```

---

## Want to write code?

**if you wanna learn how to write code, check here** → **[sharing_docs/how-to-write-code.en.md](sharing_docs/how-to-write-code.en.md)**

Covers installing/referencing both libraries, the minimal example, reusing one graph, and which way the weights go.

---

## Python: `vulkantorch`

### Run the TTS Example (`cli.py`)

[example/python/higgstts_py/cli.py](example/python/higgstts_py/cli.py): **reference audio + text → wav** (`encode_ref` + autoregressive + DAC decode).

```bat
python example\python\higgstts_py\cli.py ^
  --model path/to/HiggsTTS3-q8_0.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --ref-wav data/ref_audio/melinaref_24k.wav ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

```
=== Timing ===
Prefill:           943 ms
Backbone AR:      1273 ms
Decode:             12 ms
Total:            2227 ms
Audio:            5.72 sec
RTF:             0.389 x
```
(on the Python side, librosa init wastes 0.8 s — **only on the first call**: a second `enc` inside the same
process takes 41 ms, bringing the total to ~1300 ms / RTF 0.23, about the same as the C# side. The CLI is a
fresh process every run, so it always pays that 0.8 s.)

| Arg | Description |
|---|---|
| `--model` | **required**, the HiggsTTS GGUF; [download](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) (~4 GB) |
| `--tokenizer` | defaults to the copy in the repo, `data/ref_audio/higgs_tts_v3_tokenizer.json` |
| `--ref-wav` | reference audio, defaults to `data/ref_audio/melinaref_24k.wav` |
| `--ref-text` / `--text` | reference text / text to synthesize (style tags like `<|style:whispering|>` allowed) |
| `--out` | output wav, defaults to `data/higgstts/cli_py.wav` |
| `--temperature` `--topk` `--seed` | sampling params, default `0.9 / 50 / 42` |
| `--max-steps` | autoregressive step cap, `0` = predict from text length |
| `--encode-only` / `--no-graph-cache` | only run `encode_ref` / disable the graph cache (for A/B) |

---

## C#: `VulkanTorch.Net`

### Run the TTS Example (`HiggsTts.Net.exe`)

[example/CSharp/HiggsTts.Net/](example/CSharp/HiggsTts.Net/) is the C# counterpart of `higgstts_py`, equally self-contained (ships its own tokenizer + resampler, no Python-side export needed).

```bat
dotnet build example\CSharp\HiggsTts.Net -c Release

example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe synth ^
  --model D:\models\HiggsTTS3-q8_0.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

```
=== Timing ===
Prefill:           117 ms
Backbone AR:      1049 ms
Decode:             10 ms
Total:            1176 ms
Audio:            5.04 sec
RTF:             0.233 x
```

Two modes: `synth` (default, ref audio + text → wav) and `encode` (only `encode_ref`). Args map 1:1 to the Python version, plus `--no-graph-cache`; `-h` / `--help` prints them all.

---

## Three-Language Speed Comparison
Benchmarks in [bench/](bench/).

C# and Python use the same C++ core; only the binding layer differs.

TL;DR: "they're all about the same."

For C++ HiggsTTS I won't reinvent the wheel in this repo — just use [this project](https://github.com/Rafa00127/HiggsTTS.cpp). Same ggml-vulkan backend. The C++ version doesn't enable the graph cache, because it measured to make little difference for this case — though comparing a cached build against a non-cached C++ build is a *tiny* bit unfair.

**Single op** (ms):

| | C++ | C# | Python |
|---|---|---|---|
| matmul 4096³ | 3.75 | 3.60 | 3.72 |
| conv1d 16384 512→512 | 2.39 | 2.26 | 2.65 |

**Long-sequence autoregressive** (HiggsTTS AR, ~550 frames, ms/frame, 20+s of audio):

| | C++ (reference) | C# | Python |
|---|---|---|---|
| Rebuild graph each step | 8.12 | 8.83 | 9.42 |
| Graph cache | — | **7.46** | **7.64** |

**The graph-cached and rebuild versions produce bit-identical wavs** (not "roughly the same"), so the cache only saves time, never changes numbers.

For the long-audio synth above (~550 frames / 21.5 s audio), the RTFs: **C++ 0.207 / C# 0.195 / Python 0.240** (Python includes a one-off ~0.9 s librosa init).

---

## Example Models

Three small-model ports (two TTS + one OCR) — the code lives in `example/`, the docs in
`sharing_docs/`:

→ **[sharing_docs/example-models.en.md](sharing_docs/example-models.en.md)** — overview (summary,
speed, weight downloads)

| Model | Language | Details |
|---|---|---|
| HiggsTTS v3 | C# + Python | [sharing_docs/higgstts.en.md](sharing_docs/higgstts.en.md) |
| IndexTTS 2.5 | C# | [sharing_docs/indextts.en.md](sharing_docs/indextts.en.md) |
| PP-OCRv6 | Python | [sharing_docs/paddleocr.en.md](sharing_docs/paddleocr.en.md) |

---

## License

MIT
