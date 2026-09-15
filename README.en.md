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
| Python | 3.12 |
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

## Python: `vulkantorch`

### 1. Install

Not on PyPI. After `python build_win.py` (or `build.sh`), just add the **repo root** to `sys.path`:

```python
import sys
sys.path.insert(0, r"path/to/vulkan-torch")
import vulkantorch as vt
```

The extension bundles ggml — no DLL dir to configure.

### 2. Write Code

One conv + one linear:

```python
import numpy as np
import vulkantorch as vt

rt = vt.Runtime()
gpu = rt.gpu()

# weights on the GPU: Memory is VRAM storage; stuff tensors into it
wc = np.random.rand(64, 32, 3).astype(np.float32)      # conv   [Cout, Cin, K]
wl = np.random.rand(128, 64).astype(np.float32)        # linear [out, in]
mwc = vt.Memory(gpu, wc.nbytes).tensor([64, 32, 3], wc.tobytes())
mwl = vt.Memory(gpu, wl.nbytes).tensor([128, 64], wl.tobytes())

# one forward pass = one Graph
g = vt.Graph(rt, gpu)
g.enter()
x = g.input([64, 32], np.zeros((64, 32), np.float32).tobytes())   # [T, Cin]
h = vt.gelu(vt.conv1d(x, mwc, 1, 1, 1))                          # [T, 64]
y = vt.linear(h, mwl)                                            # [T, 128]
y.mark_output()                                                  # anything you read must be marked as output
out = np.frombuffer(y.to_bytes(), np.float32).reshape(64, 128)    # read = compute
g.exit()
```

Key points:

- Between `enter()` / `exit()` it's just **bookkeeping**; `to_bytes()` / `to_host()` is what actually computes.
- Any tensor you read must be `mark_output()`, or the scheduler will reuse its VRAM.
- `Memory` must be created **before** `enter()`; ops run on the **backend of the weights** (weights on GPU → ops on GPU).
- Ops are functions under `vt.`: `matmul` `linear` `conv1d` `rms_norm` `layer_norm` `gelu` `silu` `soft_max` `rope` `flash_attn` `snake_1d` `im2col_rafa` …
- Inputs via `g.input(shape, bytes)` (F32) / `g.input_i32(shape, bytes)`.
- **`shape()` eats the slowest dim's 1s** (at least one dim always remains):

  | PT shape | `shape()` |
  |---|---|
  | `(2, 1, 4)` | `(2, 1, 4)` |
  | `(1, 4)` | `(4,)` |
  | `(1, 2, 4)` | `(2, 4)` |

Reading weights from GGUF:

```python
f = vt.GgufFile(r"model.gguf", gpu)
w = f.tensor("some.weight")     # copies to VRAM, keeps the original quantized type
```

> float32 matmul goes through Vulkan's fp16 matrix cores, ~1e-3 off a double-precision reference — normal. It's the **device-level** default, shared by all models; to run full f32, set `GGML_VK_DISABLE_F16=1` **before** `import vulkantorch` (in C#, `SetEnvironmentVariable` before the first `vt_*` call). The `fp16:` line in the startup log shows the current state.

### 3. Reuse the Same Graph

If the shapes stay the same and only the data changes, a graph can be **allocated once**, then just re-fed new inputs — saving the repeated build + alloc inside a loop:

```python
g = vt.Graph(rt, gpu)
g.enter()
x = g.input([64, 32], np.zeros((64, 32), np.float32).tobytes())
y = vt.linear(vt.gelu(vt.conv1d(x, mwc, 1, 1, 1)), mwl)
y.mark_output()
g.exit()

g.alloc_static()                             # call once after building
for i in range(10):
    g.set_input(x, batch(i))                 # swap the input
    g.compute_static()                       # compute only, no re-alloc
    out = np.frombuffer(y.to_bytes(), np.float32).reshape(64, 128)
```

Key points:

- **`compute()` is one-shot**, only callable once; replay must use `alloc_static()` + `compute_static()`.
  (The scheduler re-allocates every time, which makes the result depend on *that* allocation pass, not the input — that's not a perf difference, it's **wrong numbers**.)
- A replayed graph must be **position-independent**: positions go in via `set_rows` as runtime inputs, attention windows pinned to a fixed size, masks as inputs too. See the bucketing in [GraphCache.cs](VulkanTorch.Net/GraphCache.cs).
- For small models, who cares; but **autoregressive** cases that re-run per generated token — there it's real money (see [Speed Comparison](#three-language-speed-comparison)).

### 4. Run the TTS Example (`cli.py`)

[example/python/higgstts_py/cli.py](example/python/higgstts_py/cli.py): **reference audio + text → wav** (`encode_ref` + autoregressive + DAC decode).

```bat
python example\python\higgstts_py\cli.py ^
  --model path/to/HiggsTTS3.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --ref-wav data/ref_audio/melinaref_24k.wav ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

```
=== Timing ===
Prefill:           933 ms
Backbone AR:      1261 ms
Decode:             10 ms
Total:            2204 ms
Audio:            5.72 sec
RTF:             0.385 x
```
(on the Python side, librosa init wastes 0.8 s)

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

### 1. Reference It

Not on NuGet. Project-reference it (or reference `VulkanTorch.Net.dll`), and drop the **native lib next to your exe**:

```xml
<ItemGroup>
  <ProjectReference Include="..\VulkanTorch.Net\VulkanTorch.Net.csproj" />
  <None Include="..\build\vulkantorch.dll" Link="vulkantorch.dll" CopyToOutputDirectory="PreserveNewest" />
</ItemGroup>
```

(ggml is statically linked in — this one file is all you need.)

### 2. Write Code

Same conv + linear:

```csharp
using VulkanTorch;

using var rt = new Runtime();
var gpu = rt.Gpu();

int T = 64, Cin = 32, Cout = 64, Lout = 128, K = 3;
var wc = new float[Cout * Cin * K];      // fill in your data (row-major)
var wl = new float[Lout * Cout];

using var mwc = new Memory(gpu, (ulong)wc.Length * 4);   // weights on the GPU
using var mwl = new Memory(gpu, (ulong)wl.Length * 4);
var twc = mwc.Tensor(new long[] { Cout, Cin, K }, Ops.F32, ToBytes(wc));
var twl = mwl.Tensor(new long[] { Lout, Cout }, Ops.F32, ToBytes(wl));

using var g = new Graph(rt, gpu);
g.Enter();
var x = g.Input(new long[] { T, Cin }, new float[T * Cin]);
var y = Ops.Linear(Ops.Gelu(Ops.Conv1d(x, twc, 1, 1, 1)), twl).MarkOutput();
var outv = y.ToFloats(g);                   // read = compute
g.Exit();

static byte[] ToBytes(float[] f)
{
    var b = new byte[f.Length * 4];
    Buffer.BlockCopy(f, 0, b, 0, b.Length);
    return b;
}
```

Same key points as Python: `Enter()`/`Exit()` is the capture scope, `ToFloats`/`ToBytes`/`ToInts` trigger compute; `MarkOutput()` anything you read; `Memory` created before `Enter()`; ops live in the `Ops` static class.

#### Which Way the Weights Go (`Linear`'s convention)

`twl` above is created as `(Lout, Cout)`, i.e. **`(out, in)`** — not a style choice, it is required:

| | Form | Weight shape (first entry is the row) |
|---|---|---|
| Textbook | `X @ W + b` | `W (in, out)` — input dim first |
| PyTorch `nn.Linear` | `x @ W.T + b` | `W (out, in)` — output dim first |
| This library | `Linear(x, w)` = `ggml_mul_mat(w, x)` = `x @ w.T` | `w (out, in)` — same as PyTorch |

**Different notation, identical memory layout**: torch's row-major and ggml's fastest-dim-first are
exact inverses, and ggml wants the contraction dim `in` at `ne0` — which is exactly where an
`(out, in)` weight already lands when read in as-is. No transpose or shuffle needed.

- Store weights as `(out, in)` and call `Ops.Linear(x, w)`
- **With a weight as `w`, don't write `Ops.Matmul(x, w)`** — it has to `cont(transpose(w))` to satisfy
  ggml's rule: wasted bandwidth on f16, a hard crash on quantized weights. `Matmul` is for activation × activation
- If upstream is an HF `Conv1D` (misleading name: it is a plain linear layer with no kernel, it just
  stores `(in, out)`), transpose it once in the converter

### 3. Reuse the Same Graph

```csharp
using var g = new Graph(rt, gpu);
g.Enter();
var x = g.Input(new long[] { T, Cin }, new float[T * Cin]);
var y = Ops.Linear(Ops.Gelu(Ops.Conv1d(x, twc, 1, 1, 1)), twl).MarkOutput();
g.Exit();

g.AllocStatic();                            // call once after building
for (int i = 0; i < 10; i++)
{
    g.SetInput(x, batch(i));                // swap the input
    g.ComputeStatic();                      // compute only, no re-alloc
    var outv = y.ToFloats(g);
}
```

Identical to the Python side: `Compute()` is one-shot, replay uses `AllocStatic()` + `ComputeStatic()`; the replayed graph must be position-independent. For autoregressive decode, just use [GraphCache.cs](VulkanTorch.Net/GraphCache.cs)'s `BucketedReplay` (build once per window bucket, auto allocate + replay) — the model only provides two callbacks: "how to build the graph" and "what's the per-frame input".

### 4. Run the TTS Example (`HiggsTts.Net.exe`)

[example/CSharp/HiggsTts.Net/](example/CSharp/HiggsTts.Net/) is the C# counterpart of `higgstts_py`, equally self-contained (ships its own tokenizer + resampler, no Python-side export needed).

```bat
dotnet build example\CSharp\HiggsTts.Net -c Release

example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe synth ^
  --model D:\models\HiggsTTS3.gguf ^
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

## Example Model: HiggsTTS

> **Weights download** — [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) (~4 GB)

Both examples port [HiggsTTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) (a 4B TTS):

```
reference audio ──► encode_ref ──► RVQ codes ──┐
                   (codec)                     ├──► Qwen3 36-layer AR ──► codes ──► DAC decode ──► wav
   text ──────────► tokenizer ──► prompt ──────┘        (backbone)                   (decoder)
```

- **Python**: `example/python/higgstts_py/` (`model.py` / `ar.py` / `decode.py` / `tts.py`)
- **C#**: `example/CSharp/HiggsTts.Net/` (`EncodeRef.cs` / `Ar.cs` / `DacDecoder.cs` / `HiggsTokenizer.cs` / `Resampler.cs`)
- **Weights**: [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) (~4 GB, not in the repo)

Numerically aligned: DAC decode and AR logits are **bit-identical**, tokenizer output matches too; with the same resampler, `encode_ref` gives 97% identical frames (the only diff comes from Python using librosa vs C# using its own Kaiser resampler — inaudible). Speed ~**0.2–0.3 RTF**.

---

## Example Model: IndexTTS 2.5

> **Weights download** — [NeemaShioSe/IndexTTS2.5.gguf](https://huggingface.co/NeemaShioSe/IndexTTS2.5.gguf) (~3.3 GB)

C#-only (lazy). Zero-shot voice cloning + emotion control, 99 languages including zh/en/ja/es.
The whole chain (text frontend → GPT-2 AR → semantic codec → s2mel/CFM → BigVGAN, plus the reference audio's fbank/Wav2Vec2-BERT/CAMPPlus) lives in [example/CSharp/IndexTts.Net/](example/CSharp/IndexTts.Net/), self-contained.

```
reference audio ─┬─ Kaldi fbank ─► CAMPPlus ──────────────────────► style
                 ├─ Kaldi fbank ─► Wav2Vec2-BERT ─► spk_cond ─┐
                 └─ 22.05k mel ───────────────────────────────┤
                                                             ▼
text ─► tokenize/normalize/segment ─► GPT-2 AR ─► codec ─► length_regulator ─► CFM(25-step Euler+CFG) ─► BigVGAN ─► wav
```

```bat
dotnet build example\CSharp\IndexTts.Net -c Release

example\CSharp\IndexTts.Net\bin\Release\net10.0\IndexTts.Net.exe synth ^
  --model model\indextts2.5\indextts2.5.f16.gguf ^
  --ref-wav data\ref_audio\melinaref_24k.wav ^
  --text "大家好，这是一个测试。" --out data\indextts\out.wav
```

| Arg | Description |
|---|---|
| `--model` | **required**, the IndexTTS 2.5 GGUF; [download](https://huggingface.co/NeemaShioSe/IndexTTS2.5.gguf) (~3.3 GB, or convert it yourself with `tools/convert/convert_index_tts2_to_gguf.py`) |
| `--ref-wav` | reference audio (the timbre source), defaults to `data/ref_audio/melinaref_24k.wav` |
| `--text` | text to synthesize; supports `<char\|pronunciation>` annotations |
| `--emo` | emotion weights, e.g. `--emo "happy=0.6,calm=0.4"` (8 kinds: happy/angry/sad/afraid/disgusted/melancholic/surprised/calm) |
| `--lang` `--out` `--seed` `--max-steps` | language / output / seed / step cap |
| `--top-p` `--top-k` `--temperature` `--rep-penalty` | sampling params (default 0.8 / 30 / 0.8 / 10) |
| `--num-beams` | beam width, default 1 |
| `--cfg-rate` `--diffusion-steps` | CFM CFG strength and diffusion steps (default 0.7 / 25) |
| `--duration-factor` | speech-rate / duration scale |
| `--no-graph-cache` | disable the AR graph cache (for A/B) |
| `-h` `--help` | print all args |

### Speed (RX 7900 XTX / Vulkan / f16 GGUF)

Single-sentence `synth`, per-stage time and RTF (RTF = inference time / audio duration, excluding weight load, same as the official):

```bat
IndexTts.Net.exe synth --model model\indextts2.5\indextts2.5.f16.gguf ^
  --ref-wav data\ref_audio\melinaref_24k.wav --text "..." --out data\indextts\out.wav
```

| Text length | Audio | GPT-2 AR | s2mel (codec+CFM) | BigVGAN | Inference total | RTF |
|---|---|---|---|---|---|---|
| 11 chars | 3.51 s | 625 ms | 631 ms | 475 ms | 1.98 s | 0.564 |
| 33 chars | 7.30 s | 1059 ms | 797 ms | 852 ms | 2.95 s | 0.404 |
| 89 chars | 19.04 s | 2384 ms | 2045 ms | 2220 ms | 6.90 s | 0.362 |

For reference, the official 2.5 RTF on an **RTX 4090** (`kv_cache=True`): bf16 ~0.20, fp32 ~0.21 (7–200 chars).
This port is roughly **1.75×** slower, mostly due to hardware (7900 XTX vs 4090) and the official using bf16 + CUDA-fused kernels, versus f16 GGUF + a generic Vulkan backend here.
For a single long sentence the three stages (AR / s2mel / BigVGAN) take almost equal time. Weight load is ~2–3 s (not counted above).

## Example Model: PP-OCRv6 (PaddleOCR)

> **Weights download** — [NeemaShioSe/paddleocr.gguf](https://huggingface.co/NeemaShioSe/paddleocr.gguf) (rec + det + dict)

Text detection + recognition, Python-only (`example/python/ocr_py/`), self-contained `.pyd`.
Usage and the precision/speed findings are in [example/python/PaddleOCR.md](example/python/PaddleOCR.md).

## Head-to-Head: the Two TTS Ports

Same machine (RX 7900 XTX / Vulkan), same 85-character Chinese sentence, same reference audio,
`--seed 42`, best of 5 with the first run discarded (RTF excludes the weight load):

| Model | Audio | Inference | RTF |
|---|---|---|---|
| HiggsTTS v3 (~4B, `higgs-v3-tts.gguf`, 9.34 GB) | 23.76 s | 6.86 s | **0.288** |
| IndexTTS 2.5 (~0.8B, `indextts2.5.f16.gguf`, 3.3 GB) | 18.40 s | 5.78 s | **0.314** |
| IndexTTS 2.5 (quantized, `indextts2.5.q8.gguf`, 2.0 GB) | 17.76 s | 5.29 s | **0.298** |

IndexTTS now beats Higgs on raw single-sentence time (5.78 vs 6.86 s). Higgs's audio came out
longer (23.76 vs 18.40 s), which is what keeps its RTF lower — RTF is sensitive to how much audio
a sentence produces, so don't read it on its own.
Higgs's `Decode` is basically free (41 ms — all the time is in the 36-layer backbone AR), while
IndexTTS splits it three ways: AR / s2mel / BigVGAN. The bench sentence and the exact commands
are in [IndexTTS2.5.en.md](example/CSharp/IndexTTS2.5.en.md).

## License

MIT
