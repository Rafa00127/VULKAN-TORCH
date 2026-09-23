# PP-OCRv6 (PaddleOCR) · Python Port

**🌏 Language / 语言:** [中文](paddleocr.md) · **English**

A from-scratch port of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s
**PP-OCRv6_medium_det / _rec** to [vulkan-torch](../README.en.md), code in
[`example/python/ocr_vt/`](../example/python/ocr_vt/),
with the ability of both recognition and detection (recognition only by default).

The whole set of ported models: [example-models.en.md](example-models.en.md).

```python
import sys
sys.path[:0] = ["<repo>", "<repo>/example/python"]   # vulkantorch, then ocr_vt
from ocr_vt.ocr import Ocr

ocr = Ocr()
text = ocr.read_line(rgb)                 # one cropped line  -> text (rec-only)
text = ocr.read_line(rgb, det=True)       # a region          -> det+rec -> text
```

One already-cropped line (or region) in, text out. Page segmentation is the caller's job.

### Some Test Results (RX 7900 XTX & 9950X)

Two inputs: 

**A** — one page of Chinese text (~2000 chars, lines pre-cut);

**B** — a chat screenshot. 

Baselines: PaddleOCR (paddlex) on this machine's CPU (same weights), and a
**PyTorch (ROCm)** build of the same models — about the speed ceiling on this GPU.

`rec` recognizes one cropped line; 

`det+rec` detects boxes first, then recognizes.

`match vk/pt` is each GPU transcript's agreement with the CPU baseline (1 − CER) — all
three run the same weights, so it's not ground-truth accuracy.

| input | mode | vulkan-torch (new input size) | vulkan-torch (cached) | PyTorch (ROCm) | CPU | match vk/pt |
|---|---|---|---|---|---|---|
| A | rec | **1.2 s** | — | 1.0 s | 16.2 s | **99.95 / 100 %** |
| A | det+rec | **2.8 s** | — | 2.3 s | 38.6 s | **98.9 / 99.0 %** |
| B | det+rec | **0.12 s** | **0.084 s** | 0.10 s | 0.8 s | **100 / 100 %** |

A is a page whose lines are pre-cut (125 lines, rec and det+rec per line); B is a screenshot
run as a whole-image det+rec.

vulkan-torch builds a compute graph on every new input size, which costs a little extra
versus eager PyTorch. In fixed-shape recognition — video subtitles, galgame dialogue, anything
where the crop size repeats — there is **no gap** (B: 0.084 s vs torch 0.10 s).

**7–14× faster than the CPU pipeline** in every case. The trade is that vulkan-torch runs on
any Vulkan GPU with **no ROCm or CUDA**.

Where any of them differ it's a few punctuation/quote glyphs, not content.

### Weights


Get them either way:

- **Prebuilt** — [NeemaShioSe/paddleocr.gguf](https://huggingface.co/NeemaShioSe/paddleocr.gguf) on HuggingFace
  (rec + det + dict)

- **Convert locally** — `python tools/convert/convert_ocr_to_gguf.py`

  (reads PaddleX's own safetensors export under `~/.paddlex/official_models/`;
  folds BatchNorm into the convs). `--outtype f16` halves the file but is **not
  recommended** — see the f16 caveat below.

**Models don't have to live in the repo** — point the library at them:

```python
ocr = Ocr(model_dir="D:/models/ppocrv6")          # any dir with the standard filenames
ocr = Ocr(rec_path="...", det_path="...", dict_path="...")   # or per-file
```

or the env vars `OCR_MODEL_DIR` / `OCR_PRECISION` / `OCR_REC_GGUF` / `OCR_DET_GGUF`
/ `OCR_DICT`. `OCR_PRECISION` selects the GGUF for **both** rec and det (`f32` default;
`f16` falls back to `f32` for whichever model wasn't converted as f16). rec-only needs
just the rec GGUF + dict (det loads lazily on the first `det=True`).

### Files

```
ocr_vt/
  ocr.py        Ocr API (read_line) + det/rec preprocess
  rec.py        LCNetV4 + EncoderWithLightSVTR + CTC head
  det.py        LCNetV4 + RepLKFPN neck + DB head
  backbone.py   shared PP-LCNetV4
  postproc.py   DB box postprocess + CTC decode
  weights.py    GGUF -> device tensors
  nn.py         op helpers (conv+BN-fold bias, hardsigmoid, layer_norm, pools)
  cli.py        python example/python/ocr_vt/cli.py --line line.png
  screen_translator.py  PyQt6 screen 划词翻译 tool: drag a box over on-screen text -> OCR -> (optional) LLM translate
```

### Screen translator tool

[`screen_translator.py`](../example/python/ocr_vt/screen_translator.py) is a small PyQt6 desktop app built on the
port: hit the hotkey (default `Ctrl+Alt+Shift+O`, customizable) or click the button, drag a box
over any on-screen text, and it OCRs the box (rec-only, or det+rec for multi-line) and — if 翻译
is ticked — translates it via an OpenAI-compatible LLM endpoint (base/model/key in the config).
Handles vertical (manga) text automatically (rotates tall boxes 90°). UI language can be
switched (中文/English) and the choice is saved.

```bat
python example\python\ocr_vt\screen_translator.py
```

Settings (hotkey, LLM endpoint, UI language, …) live in `data/ocr/screen_translator.json`
(gitignored) and are editable from the 「设置」 dialog.

### Notes

- **rec-only is the default** — `det=True` runs the full det+rec on a region and returns
  newline-joined text.
- **Default F32 — keep it.** More accurate, and f16 is not faster.
- **CTC is decoded on the GPU.**
- **det matches PaddleOCR.**
- **Convolutions run on matrix cores** (2026-09): the fused conv path is ~2.1× faster on
  det's 9×9 kernels. Mind the measurement setup if you re-run it: weights pre-loaded, line
  segmentation outside the timing, and A/B interleaved.
