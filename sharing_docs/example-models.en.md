# Example Models

**🌏 Language / 语言:** [中文](example-models.md) · **English**

The repo ports four small models (two TTS, two OCR) using
[vulkantorch / VulkanTorch.Net](../README.en.md). Each gets one short paragraph and the
findings here; **the details live in each model's own md**.

## Contents

| Model | Ported in | Details |
|---|---|---|
| HiggsTTS v3 — a 4B TTS | C# + Python | [higgstts.en.md](higgstts.en.md) |
| IndexTTS 2.5 — zero-shot cloning + emotion | C# | [indextts.en.md](indextts.en.md) |
| PP-OCRv6 — text detection + recognition | Python | [paddleocr.en.md](paddleocr.en.md) |
| PaddleOCR-VL 1.6 — whole-page OCR / tables / formulas | Python | [paddleocrvl.en.md](paddleocrvl.en.md) |

---

## HiggsTTS v3

> - **Details** — **[higgstts.en.md](higgstts.en.md)**
> - **Weights** — [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) (~4 GB)
> - **Upstream** — [bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)

HiggsTTS (4B) is ported once per library:
Python in [`example/python/higgstts_vt/`](../example/python/higgstts_vt/), C# in
[`example/CSharp/HiggsTtsSharp/`](../example/CSharp/HiggsTtsSharp/) (CLI in `HiggsTts.Net/`),
both self-contained.
Reference audio → RVQ codes → stuffed into the prompt → Qwen3 36-layer AR → DAC decode.

The two implementations are **bit-identical** (DAC decode and AR logits match), speed is
**0.2–0.3 RTF**, and nearly all of it goes into the 36-layer backbone AR.

---

## IndexTTS 2.5

> - **Details** — **[indextts.en.md](indextts.en.md)**
> - **Weights** — [NeemaShioSe/IndexTTS2.5.gguf](https://huggingface.co/NeemaShioSe/IndexTTS2.5.gguf) (~3.3 GB)
> - **Upstream** — [index-tts/index-tts](https://github.com/index-tts/index-tts)

A C# port of IndexTTS 2.5 (zero-shot voice cloning + emotion control, 99 languages including
zh/en/ja/es). The whole chain (text frontend → GPT-2 AR → semantic codec → s2mel/CFM → BigVGAN,
plus the reference audio's fbank/Wav2Vec2-BERT/CAMPPlus) lives in
[`example/CSharp/IndexTtsSharp/`](../example/CSharp/IndexTtsSharp/) (CLI in `IndexTts.Net/`),
self-contained.

`--emo` blends 8 emotions and `--text` accepts `<char|pronunciation>` annotations;
**RTF 0.314** (f16) / **0.298** (q8).

---

## PP-OCRv6 (PaddleOCR)

> - **Details** — **[paddleocr.en.md](paddleocr.en.md)**
> - **Weights** — [NeemaShioSe/paddleocr.gguf](https://huggingface.co/NeemaShioSe/paddleocr.gguf) (rec + det + dict)
> - **Upstream** — [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

A port of PaddleOCR's **PP-OCRv6_medium_det / _rec**, Python-only
([`example/python/ocr_vt/`](../example/python/ocr_vt/)), self-contained `.pyd`. A cropped line (or
a whole region) in, text out — page segmentation is the caller's job.

Recognition is rec-only by default; `det=True` detects boxes first. Transcripts agree with the
CPU baseline **99.95%** of the time, it's **7–14× faster than the CPU pipeline**, and at a fixed
input size it matches — or even outspeeds — PyTorch (ROCm) a tiny little bit on my device
(0.084 s vs 0.10 s).

---

## PaddleOCR-VL 1.6

> - **Details** — **[paddleocrvl.en.md](paddleocrvl.en.md)**
> - **Weights** — [PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/tree/main) (model + mmproj)
> - **Upstream** — [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

A 0.3B **vision-language model** (ERNIE-4.5-0.3B plus a modified SigLIP tower), Python-only
([`example/python/paddleocrvl_vt/`](../example/python/paddleocrvl_vt/)). Image plus a task prefix
(`OCR:` / `Table Recognition:` / ...) in, structured text out.

Output is **byte-identical** to a llama-server on llama.cpp (ROCm); a 937-token screenshot takes
0.77 s (the vision tower is 0.46 s of that, decode is 482 tok/s steady-state).
