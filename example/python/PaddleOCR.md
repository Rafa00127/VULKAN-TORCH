# Python examples (vulkan-torch)

**🌏 Language / 语言:** [中文](PaddleOCR.zh-CN.md) · **English**

| dir | model |
|---|---|
| [`ocr_py/`](ocr_py/) | PP-OCRv6 text detection + recognition (PaddleOCR port) |
| [`higgstts_py/`](higgstts_py/) | HiggsTTS (reference voice → speech) |

---

## `ocr_py/` — PaddleOCR (PP-OCRv6) on vulkan-torch

A from-scratch port of PaddleOCR's **PP-OCRv6_medium_det / _rec** to vulkan-torch,
built for a cropped line (recognition only by default).

```python
import sys
sys.path[:0] = ["<repo>", "<repo>/example/python"]   # vulkantorch, then ocr_py
from ocr_py.ocr import Ocr

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

| input | mode | vulkan-torch | PyTorch (ROCm) | CPU | match vk/pt |
|---|---|---|---|---|---|
| A | rec | **1.2 s** | 1.0 s | 16.2 s | **99.95 / 100 %** |
| A | det+rec | **3.4 s** | 2.3 s | 38.6 s | **98.9 / 99.0 %** |
| B | det+rec | **0.11 s** | 0.10 s | 0.8 s | **100 / 100 %** |

**7–14× faster than the CPU pipeline**, and **~1.2–1.5× slower than the ROCm build** — the
gap is almost entirely detection (convolutions run 2–3× behind there); recognition is near
parity. The trade is that vulkan-torch runs on any Vulkan GPU with **no ROCm or CUDA**.

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
ocr_py/
  ocr.py        Ocr API (read_line) + det/rec preprocess
  rec.py        LCNetV4 + EncoderWithLightSVTR + CTC head
  det.py        LCNetV4 + RepLKFPN neck + DB head
  backbone.py   shared PP-LCNetV4
  postproc.py   DB box postprocess + CTC decode
  weights.py    GGUF -> device tensors
  nn.py         op helpers (conv+BN-fold bias, hardsigmoid, layer_norm, pools)
  cli.py        python example/python/ocr_py/cli.py --line line.png
```

### Notes

- **rec-only is the default** — much faster and enough when lines are pre-cropped.
  `det=True` runs full det+rec on a region and returns newline-joined text.
- **Default F32 — keep it.** F16 halves the weights (det 88→44 MB, rec 76→38 MB) but
  **is not faster** (det+rec is conv-bound, not bandwidth-bound; measured identical on
  the test images) and it's **slightly less accurate**: f16 det crops are a touch rougher
  and f16 rec emits a few spurious glyphs (e.g. on the QQ screenshot f32 gave 118 clean
  chars, f16 gave 119 including 3 `�`). Use `OCR_PRECISION=f16` only if you're
  short on VRAM.
- **CTC is decoded on the GPU** (`rec.forward_ids` → `mt.argmax`): the graph emits T
  int32 ids instead of the T×18710 softmax, so the ~17 MB/line logits never cross PCIe
  (that transfer was ~half the runtime). Ids are bit-identical to argmax-ing the
  softmax. 14.1 → 9.6 ms/line on the test frame.
- **det matches PaddleOCR.** The OCR *pipeline* overrides the model's own
  `inference.yml`, so use the pipeline's values, not the model's:
  - preprocess `limit_side_len=64` (not 736) — `paddlex/configs/pipelines/OCR.yaml`;
  - postprocess `thresh=0.3, box_thresh=0.6, unclip=1.5` (not 0.2/0.45/1.4) plus
    `SortQuadBoxes` (top→bottom, left→right).
  The small input also made det cheap: the old 224×4000 input needed `conv2d_tiled` to
  dodge Vulkan's 2 GiB single-buffer cap; at 64×N it doesn't. These two bugs together
  were why det+rec scored 74 % before.
- BatchNorms are folded at conversion; the graph mirrors PaddleOCR layer for layer.
- Needs the ops added in `src/ops.cpp`. No install — self-contained `.pyd`, just `sys.path`.
