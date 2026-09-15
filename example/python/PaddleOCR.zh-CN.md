# Python 示例（vulkan-torch）

**🌏 Language / 语言:** **中文** · [English](PaddleOCR.md)

| 目录 | 模型 |
|---|---|
| [`ocr_py/`](ocr_py/) | PP-OCRv6 文本检测 + 识别（PaddleOCR 移植） |
| [`higgstts_py/`](higgstts_py/) | HiggsTTS（参考音色 → 语音） |

---

## `ocr_py/` —— PaddleOCR（PP-OCRv6）跑在 vulkan-torch 上

从零把 PaddleOCR 的 **PP-OCRv6_medium_det / _rec** 移植到 vulkan-torch，
面向已裁好的单行（默认只做识别）。

```python
import sys
sys.path[:0] = ["<repo>", "<repo>/example/python"]   # vulkantorch，然后是 ocr_py
from ocr_py.ocr import Ocr

ocr = Ocr()
text = ocr.read_line(rgb)                 # 一行裁好的图 -> 文本（只识别）
text = ocr.read_line(rgb, det=True)       # 一块区域      -> det+rec -> 文本
```

输入是裁好的单行（或一块区域），输出文本。整页怎么切行是调用方的事。

### 一些测试结果（RX 7900 XTX & 9950X）

两个输入：

**A** —— 一整页中文（约 2000 字，已切好行）；

**B** —— 一张聊天截图。

基准：本机 CPU 上的 PaddleOCR（paddlex，同一套权重），以及同一批模型的
**PyTorch（ROCm）**构建——大致是这张卡的速度上限。

`rec` 识别一行裁好的图；

`det+rec` 先检测框、再逐行识别。

`match vk/pt` 是每个 GPU 结果与 CPU 基准的吻合度（1 − CER）——三者跑的是同一套权重，
所以不是真值准确率。

| 输入 | 模式 | vulkan-torch（新输入尺寸） | vulkan-torch（缓存命中） | PyTorch (ROCm) | CPU | match vk/pt |
|---|---|---|---|---|---|---|
| A | rec | **1.2 s** | — | 1.0 s | 16.2 s | **99.95 / 100 %** |
| A | det+rec | **3.1 s** | — | 2.3 s | 38.6 s | **98.9 / 99.0 %** |
| B | det+rec | **0.14 s** | **0.10 s** | 0.10 s | 0.8 s | **100 / 100 %** |

vulkan-torch 每换一个输入尺寸就要建一次计算图，比 eager 的 PyTorch 多付一点这个开销。
固定形状的识别——视频字幕、galgame 台词、任何裁剪尺寸会重复的场景——**没有差距**
（B：0.10 s vs torch 0.10 s）。

**任何情况下都比 CPU 流水线快 7–14×**。换来的是 vulkan-torch 能在任何 Vulkan 卡上跑，
**不需要 ROCm 或 CUDA**。

哪里不同也就是几个标点/引号字形，内容没差。

### 权重


两种方式都行：

- **预编译** —— HuggingFace 上的 [NeemaShioSe/paddleocr.gguf](https://huggingface.co/NeemaShioSe/paddleocr.gguf)
  （rec + det + dict）

- **本地转换** —— `python tools/convert/convert_ocr_to_gguf.py`

  （读 PaddleX 自己导出的 safetensors，在 `~/.paddlex/official_models/` 下；
  把 BatchNorm 折进卷积）。`--outtype f16` 把文件减半但**不推荐**——见下面的 f16 提醒。

**模型不必放在仓库里** —— 让库去指：

```python
ocr = Ocr(model_dir="D:/models/ppocrv6")          # 任何含标准文件名的目录
ocr = Ocr(rec_path="...", det_path="...", dict_path="...")   # 或逐个文件指定
```

或者用环境变量 `OCR_MODEL_DIR` / `OCR_PRECISION` / `OCR_REC_GGUF` / `OCR_DET_GGUF`
/ `OCR_DICT`。`OCR_PRECISION` **对 rec 和 det 同时生效**（默认 `f32`；
`f16` 时哪个模型没转 f16 就回退 f32）。只做 rec 只需要 rec 的 GGUF + 字典
（det 在第一次 `det=True` 时才懒加载）。

### 文件

```
ocr_py/
  ocr.py        Ocr API（read_line）+ det/rec 预处理
  rec.py        LCNetV4 + EncoderWithLightSVTR + CTC head
  det.py        LCNetV4 + RepLKFPN neck + DB head
  backbone.py   共用的 PP-LCNetV4
  postproc.py   DB 框后处理 + CTC 解码
  weights.py    GGUF -> 设备张量
  nn.py         算子助手（conv+BN 折偏置、hardsigmoid、layer_norm、pool）
  cli.py        python example/python/ocr_py/cli.py --line line.png
  screen_translator.py  PyQt6 屏幕划词翻译工具：框选屏幕文字 -> OCR ->（可选）LLM 翻译
```

### 屏幕划词翻译工具

[`screen_translator.py`](ocr_py/screen_translator.py) 是基于本小模型的一个小 PyQt6 桌面工具：
按快捷键（默认 `Ctrl+Alt+Shift+O`，可自定义）或点按钮，在屏幕上框选任意文字，它会识别框内内容
（默认只识别；勾「多行/整块」走 det+rec）——勾上「翻译」再用 OpenAI 兼容的 LLM 端点翻译
（base/model/key 都在 config 里）。竖排（漫画）文字自动处理（瘦高的框转 90°）。界面语言可切换
（中文/English）且会记住。

```bat
python example\python\ocr_py\screen_translator.py
```

设置（快捷键、LLM 端点、界面语言……）存在 `data/ocr/screen_translator.json`（已 gitignore），
可在「设置」对话框里改。

### 说明

- **默认只做 rec** —— `det=True` 在一块区域上跑完整 det+rec，返回换行拼接的文本。
- **默认 F32** —— 更准，而且 f16 并不会更快。
- **CTC 在 GPU 上解码**
- **det 与 PaddleOCR 对齐。** 
