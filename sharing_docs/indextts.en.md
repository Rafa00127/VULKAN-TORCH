# IndexTTS 2.5 · C# Port

**🌏 Language / 语言:** [中文](indextts.md) · **English**

A C# port of [IndexTTS 2.5](https://github.com/index-tts/index-tts) (Bilibili; zero-shot voice
cloning + emotion control) on top of [vulkan-torch](../README.en.md). The whole inference
chain is self-contained — nothing needs to be exported from Python.

```
reference audio ─┬─ fbank ─► CAMPPlus ─────────────────► style ──────────────┐
                 ├─ fbank ─► Wav2Vec2-BERT ─► spk_cond ─┐                     │
                 └─ 22.05k mel ────────────────────────┐│                     │
                                                       ▼▼                     ▼
text ─► tokenize/normalize/segment ─► GPT-2 AR ─► codec ─► length reg ─► CFM(25-step Euler+CFG) ─► BigVGAN ─► wav
```

(`prompt_condition` and s2mel's `cond` both come out of that length regulator: one stretches the
reference audio's `spk_cond` to `ref_mel`'s length, the other stretches the codec output to the target duration.)

---

## 1. Prepare the Model

You need a GGUF (~3.3GB) and a tiktoken vocab. The GGUF is converted from the official 4 weight files:

```bat
python tools\convert\convert_index_tts2_to_gguf.py --model <original-weights-dir> ^
  --out model\indextts2.5\indextts2.5.f16.gguf
```

(Original weights: IndexTTS 2.5's `gpt.pth` / `codec.pth` / `s2mel.pth`, plus `w2v/` `camplus/` `bigvan/`.
The vocab `multilingual_zh_ja_yue_char_del.tiktoken` already ships in-repo, under `model/indextts2.5/`.)

---

## 2. Command Line

```bat
dotnet build example\CSharp\IndexTts.Net -c Release

example\CSharp\IndexTts.Net\bin\Release\net10.0\IndexTts.Net.exe synth ^
  --model model\indextts2.5\indextts2.5.f16.gguf ^
  --ref-wav data\ref_audio\melinaref_24k.wav ^
  --text "大家好，这是一个测试。" ^
  --out data\indextts\out.wav
```

| Arg | Description |
|---|---|
| `--model` | **required**, the IndexTTS 2.5 GGUF |
| `--tokenizer` | tiktoken vocab, defaults to `model/indextts2.5/multilingual_zh_ja_yue_char_del.tiktoken` |
| `--ref-wav` | reference audio (the timbre source), defaults to `data/ref_audio/melinaref_24k.wav` |
| `--text` | text to synthesize |
| `--out` | output wav, defaults to `data/indextts/cli.wav` |
| `--emo` | emotion weights, e.g. `--emo "happy=0.6,calm=0.4"` |
| `--lang` | language code, default `zh` (99 languages incl. zh/en/ja/es) |
| `--num-beams` | beam width, default `1` |
| `--top-p` `--top-k` `--temperature` `--rep-penalty` | sampling params, default `0.8 / 30 / 0.8 / 10` |
| `--cfg-rate` `--diffusion-steps` | CFM CFG strength and diffusion steps, default `0.7 / 25` |
| `--duration-factor` | speech-rate / duration scale, default `1.0` |
| `--seed` `--max-steps` | seed / autoregressive step cap (`0` = from text length) |
| `--no-graph-cache` | disable the AR graph cache (for A/B) |

**Emotion** has 8 kinds; the weights are auto-biased and capped at a sum of 0.8:

```
happy  angry  sad  afraid  disgusted  melancholic  surprised  calm
```

**Pronunciation annotations** go straight in `--text`, e.g. `今天天气<好|HAO3>不错。`

---

## 3. Using It in Code

Each stage is its own class; inputs/outputs are uniformly **PyTorch row-major**, activations as
`PT [T, C]` (time first). All ops must be called between a `Graph`'s `Enter()`/`Exit()`; reading the
result is what computes.

Below is just the **skeleton** (graph-building and parameter details elided at `/* ... */`); the full
runnable wiring is in [Cli.cs](../example/CSharp/IndexTts.Net/Cli.cs).

```csharp
using var rt = new Runtime();
var dev = rt.Gpu();
using var w = new GgufWeights(ggufPath, dev,
    new[] { "gpt.", "codec.", "s2mel.", "w2v.", "campplus.", "bigvgan.", "emo." });

// text -> segmented token ids
var tok = new IndexTokenizer(tokenizerPath);
var front = new TextFrontend(tok);
var enc = front.EncodeForInference(text, 120, "zh");   // enc.Segments / enc.SegmentTokenIds

// reference audio -> timbre / emotion conditioning (both rely on the host-side DSP frontend)
var fb = Dsp.KaldiFbank(ToDouble(Resampler.Resample(wav, sr, 16000)), 16000, 80);
var feats = Dsp.Stack((float[,])fb.Clone(), out var mask);
var style = /* CampPlus.Forward(...) */
var spk   = /* W2vBert.Forward(feats, mask, mean, std) */

// AR generates mel codes
using var ar = new ArDecoder(rt, dev, w, 2048);
var logits = ar.Prefill(conds, ids, langId);           // conds = [3,1280] soft prefix
var codes  = ar.Generate(logits, ar.PromptLen, maxNew, rng);

// codes -> semantic features -> s2mel conditioning -> CFM -> waveform
var sinfer = new SemanticCodec(dev, w).Decode(g, codesT);          // [2T, 1024]
var cond   = new LengthRegulator(w).Forward(g, sinfer, targetLen); // [L, 512]
var mel    = new Cfm(rt, dev, dit).Inference(z, refMel, catCond, style, refMelLen, 0.7f, 25);
var wav    = new BigVgan(dev, w).Forward(g, melT);                 // [L*256, 1]
```

The complete wiring is [Cli.cs](../example/CSharp/IndexTts.Net/Cli.cs) — a readable end-to-end example.

| Stage | File | Entry |
|---|---|---|
| tiktoken tokenize | [IndexTokenizer.cs](../example/CSharp/IndexTtsSharp/IndexTokenizer.cs) | `Encode(text)` |
| normalize / segment / pron. | [TextNormalization.cs](../example/CSharp/IndexTtsSharp/TextNormalization.cs) · [TextFrontend.cs](../example/CSharp/IndexTtsSharp/TextFrontend.cs) | `EncodeForInference(...)` |
| host DSP (FFT / fbank / mel) | [Dsp.cs](../example/CSharp/IndexTtsSharp/Dsp.cs) | `KaldiFbank` `Stack` `MelSpectrogram` |
| timbre / style | [CampPlus.cs](../example/CSharp/IndexTtsSharp/CampPlus.cs) | `Forward(g, fbank)` |
| semantic encoder | [W2vBert.cs](../example/CSharp/IndexTtsSharp/W2vBert.cs) | `Forward(g, feats, mask, mean, std)` |
| emotion conditioning | [EmoCond.cs](../example/CSharp/IndexTtsSharp/EmoCond.cs) · [EmotionVector.cs](../example/CSharp/IndexTtsSharp/EmotionVector.cs) | `GetEmovec` / `Blend` |
| GPT-2 autoregressive | [ArDecoder.cs](../example/CSharp/IndexTtsSharp/ArDecoder.cs) · [BeamAr.cs](../example/CSharp/IndexTtsSharp/BeamAr.cs) | `Prefill` / `Generate` |
| semantic codec | [SemanticCodec.cs](../example/CSharp/IndexTtsSharp/SemanticCodec.cs) | `Decode(g, codes)` |
| duration regulator | [LengthRegulator.cs](../example/CSharp/IndexTtsSharp/LengthRegulator.cs) | `Forward(g, x, ylens)` |
| diffusion (s2mel) | [Dit.cs](../example/CSharp/IndexTtsSharp/Dit.cs) · [Cfm.cs](../example/CSharp/IndexTtsSharp/Cfm.cs) | `Cfm.Inference(...)` |
| vocoder | [BigVgan.cs](../example/CSharp/IndexTtsSharp/BigVgan.cs) | `Forward(g, mel)` |

### Accuracy

Weights are f16, matmuls go through fp16 cores. Per-stage relative error (vs official PyTorch):
codec 2.3e-2, s2mel 5–8e-3, BigVGAN 3.8e-2, Wav2Vec2-BERT 2.0e-2.

Speed (7900 XTX, `--seed 42`, best of 5 with the first run discarded): **RTF 0.314** — 5.78 s of
inference for 18.40 s of audio, plus a 3.5 s weight load on a cold start. The quantized GGUF
(`indextts2.5.q8.gguf`) runs at **RTF 0.298** with the weight load down to 2.8 s, at the cost of
mel_logits relative error 3.3e-2 → 4.8e-2 (argmax agreement 24/24 → 21/24).

The bench sentence is 85 Chinese characters:
「大家好，这是一段用来测试语音合成速度的文字，长度大概九十个汉字。我们想比较两个模型在同一台机器、同一句话下各自的推理耗时和实时率，所以句子要足够长，长到能体现出差异，同时也要读起来自然。」

---

## 4. TODO

- **Arabic numerals in Japanese / Spanish read wrong.** The reference routes those two languages
  through NeMo's WFST text normalization (compiled OpenFST grammars, not portable) — we didn't do it.
  Chinese and English numerals are normalized; those are fine.
- Emotion reference audio (the reference's `emo_audio_prompt`) only implements the default path
  ("no emotion audio → use the speaker audio").
- Not done: text→emotion (needs the extra QwenEmotion model), streaming output.
- `--num-beams 3` can pick an ultra-short sequence under emotion conditioning (HF scores beams by the
  raw sum of log-probs, which favors short sequences).
