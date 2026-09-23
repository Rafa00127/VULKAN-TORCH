# PaddleOCR-VL 1.6 · Python Port

**🌏 Language / 语言:** [中文](paddleocrvl.md) · **English**

A port of [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) to
[vulkan-torch](../README.en.md): a 0.3B vision-language model that takes an image plus a task
prefix and emits text directly (whole-page OCR, tables, formulas, charts, seals, spotting).
Code in [`example/python/paddleocrvl_vt/`](../example/python/paddleocrvl_vt/).

Different from [PP-OCRv6](paddleocr.en.md). All ported models:
[example-models.en.md](example-models.en.md).

```bash
python example/python/paddleocrvl_vt/cli.py page.png -t ocr
```

```python
import sys
sys.path[:0] = ["<repo>", "<repo>/example/python"]   # vulkantorch, then paddleocrvl_vt
from paddleocrvl_vt import PaddleOCRVL

vl = PaddleOCRVL("PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf")
text, n_prompt, n_gen = vl.generate_image("page.png", prompt="OCR:")
```

Tasks are **literal prefixes**: `OCR:` / `Table Recognition:` / `Formula Recognition:` /
`Chart Recognition:` / `Seal Recognition:` / `Spotting:` (`cli.py -t ocr` is the short form, `-p`
takes any prefix). Greedy sampling only.

### Some Test Results (RX 7900 XTX)

A 1842x388 chat screenshot, against a llama-server on llama.cpp (ROCm):

| | vulkan-torch | llama-server |
|---|---|---|
| image -> 924 image tokens (ViT + projector) | 458 ms | inside prefill |
| LLM prefill (T=937) | 56 ms | 472 ms |
| decode | **2.07 ms/tok** (482 tok/s) | 1.85 ms/tok |
| the whole screenshot (937 in + 109 out) | **0.77 s** | 0.67 s |

The **vision tower** is the slow part (0.46 s). **Output is byte-identical to llama-server**; to
check:

```bash
python tools/vl_compare.py <image> -t ocr --model ... --mmproj ...   # token counts + text diff
```

### Weights

Both files are needed (language model + vision tower); the default directory is
`<repo>/model/paddleocrvl/`. To point elsewhere:

```bash
python example/python/paddleocrvl_vt/cli.py page.png --model D:/m/vl.gguf --mmproj D:/m/vl-mm.gguf
set PADDLEOCRVL_MODEL_DIR=D:/models/paddleocrvl     # a dir holding those two filenames
```

Dependencies are just **numpy + Pillow** (the vocabulary ships in the GGUF and the SPM merge is
implemented in [`tokenizer.py`](../example/python/paddleocrvl_vt/tokenizer.py); no sentencepiece).

### Files

```
paddleocrvl_vt/
  vl.py         PaddleOCRVL: orchestration + greedy decode + KV cache
  vision.py     SigLIP ViT (27 layers) + 2x2 patch merge + projector
  llm.py        ERNIE-4.5-0.3B (18 layers, M-RoPE, no biases, tied lm_head)
  preproc.py    smart-resize to a multiple of 28, normalize, position-embedding resize
  tokenizer.py  SPM tokenizer
  gguf_meta.py  GGUF header KV reader
  cli.py        python example/python/paddleocrvl_vt/cli.py page.png -t ocr
  screen_translator.py  PyQt6 screen tool: select text -> read it -> (optional) LLM translation
```

### Screen translator

```bash
python example/python/paddleocrvl_vt/screen_translator.py
```

Draw a box over any on-screen text -> read it with the current task prefix -> (optionally) LLM
translate it; the result can be shown as a pinned subtitle (click-through). Hotkey, task, model
dir and LLM endpoint all live in Settings, stored at `%APPDATA%/screen-translator-vl/config.json`.
The UI shell is shared with [PP-OCRv6's tool](paddleocr.en.md)
([`example/python/screen_translator_core/`](../example/python/screen_translator_core/)). Needs PyQt6.

### Notes

- **Feeding a whole page degenerates** (llama-server degenerates into repeated characters on the
  same input). The working envelope is roughly **<=2500 image tokens, <=2750 px tall**; cutting
  the page into strips is the caller's job.
- The image cap ships **inside the mmproj** (`clip.vision.image_max_pixels`, ~1.0 MPx stock);
  anything larger is **silently downscaled**.
