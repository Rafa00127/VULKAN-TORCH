# PaddleOCR-VL 1.6 · Python 移植

**🌏 Language / 语言:** **中文** · [English](paddleocrvl.en.md)

把 [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) 移植到 [vulkan-torch](../README.md)：
0.3B 的视觉语言模型，喂图 + 任务前缀，直接出文本（整页 OCR、表格、公式、图表、印章、定位）。
代码在 [`example/python/paddleocrvl_vt/`](../example/python/paddleocrvl_vt/)。

和 [PP-OCRv6](paddleocr.md) 不一样。所有示例模型：[example-models.md](example-models.md)。

```bash
python example/python/paddleocrvl_vt/cli.py page.png -t ocr
```

```python
import sys
sys.path[:0] = ["<repo>", "<repo>/example/python"]   # vulkantorch，然后是 paddleocrvl_vt
from paddleocrvl_vt import PaddleOCRVL

vl = PaddleOCRVL("PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf")
text, n_prompt, n_gen = vl.generate_image("page.png", prompt="OCR:")
```

任务是**字面前缀**：`OCR:` / `Table Recognition:` / `Formula Recognition:` / `Chart Recognition:`
/ `Seal Recognition:` / `Spotting:`（`cli.py -t ocr` 是简写，`-p` 收任意前缀）。只做贪心采样。

### 测试结果（RX 7900 XTX）

一张 1842×388 的聊天截图，对照 llama.cpp(ROCm) 的 llama-server：

| | vulkan-torch | llama-server |
|---|---|---|
| 图像 → 924 个图像 token（ViT + 投影头） | 458 ms | 含在 prefill |
| LLM prefill（T=937） | 56 ms | 472 ms |
| 解码 | **2.07 ms/tok**（482 tok/s） | 1.85 ms/tok |
| 整张（937 入 + 109 出） | **0.77 s** | 0.67 s |

慢在**视觉塔**（0.46 s）。**输出与 llama-server 逐字一致**；对拍：

```bash
python tools/vl_compare.py <图> -t ocr --model ... --mmproj ...   # 打印两边 token 数与文本 diff
```

### 权重

从 **[PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/tree/main)**
下载**两个**文件（语言模型 + 视觉塔）：

- **`PaddleOCR-VL-1.6-GGUF.gguf`** —— ERNIE-4.5-0.3B + 词表
- **`PaddleOCR-VL-1.6-GGUF-mmproj.gguf`** —— SigLIP 视觉塔 + 投影头

默认找 `<repo>/model/paddleocrvl/`。指到别处：

```bash
python example/python/paddleocrvl_vt/cli.py page.png --model D:/m/vl.gguf --mmproj D:/m/vl-mm.gguf
set PADDLEOCRVL_MODEL_DIR=D:/models/paddleocrvl     # 目录里有那俩文件名即可
```

依赖只有 **numpy + Pillow**（词表在 GGUF 里，SPM 分词在 [`tokenizer.py`](../example/python/paddleocrvl_vt/tokenizer.py)
里自己实现了，不需要 sentencepiece）。

### 文件

```
paddleocrvl_vt/
  vl.py         PaddleOCRVL：编排 + 贪心解码 + KV cache
  vision.py     SigLIP ViT（27 层）+ 2x2 patch merge + 投影头
  llm.py        ERNIE-4.5-0.3B（18 层，M-RoPE，无 bias，lm_head 绑定）
  preproc.py    smart-resize 对齐 28、归一化、位置嵌入插值
  tokenizer.py  SPM 分词
  gguf_meta.py  读 GGUF 头部 KV
  cli.py        python example/python/paddleocrvl_vt/cli.py page.png -t ocr
  screen_translator.py  PyQt6 划词工具：框选屏幕文字 -> 识别 ->（可选）LLM 翻译
```

### 屏幕划词工具

```bash
python example/python/paddleocrvl_vt/screen_translator.py
```

框选屏幕任意文字 → 按当前「任务」前缀识别 →（可选）LLM 翻译，结果可显示成置顶悬浮字幕
（鼠标穿透）。快捷键/任务/模型目录/LLM 端点都在设置里，存在
`%APPDATA%/screen-translator-vl/config.json`。界面外壳和 [PP-OCRv6 那个](paddleocr.md)共用
（[`example/python/screen_translator_core/`](../example/python/screen_translator_core/)）。需要 PyQt6。

### 注意

- **整页直喂会退化**（llama-server 同样退化成重复字符）。工作范围大约 **≤2500 图像 token、
  高 ≲2750 px**，切长条是调用方的事。
- 图像上限写在 **mmproj 里**（`clip.vision.image_max_pixels`，官方约 1.0 MPx），超了**静默缩小**。
