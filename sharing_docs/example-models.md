# 示例模型

**🌏 Language / 语言:** **中文** · [English](example-models.en.md)

仓库里用 [vulkantorch / VulkanTorch.Net](../README.md) 移植了四个小模型（两个 TTS、两个 OCR）。
这里每个只写一段简介和结论，**详细内容在各自的 md 里**。

## 目录

| 模型 | 移植的语言 | 详细 |
|---|---|---|
| HiggsTTS v3 — 4B TTS | C# + Python | [higgstts.md](higgstts.md) |
| IndexTTS 2.5 — 零样本克隆 + 情绪 | C# | [indextts.md](indextts.md) |
| PP-OCRv6 — 文本检测 + 识别 | Python | [paddleocr.md](paddleocr.md) |
| PaddleOCR-VL 1.6 — 整页 OCR / 表格 / 公式 | Python | [paddleocrvl.md](paddleocrvl.md) |

---

## HiggsTTS v3

> - **详细文档** —— **[higgstts.md](higgstts.md)**
> - **权重** —— [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf)（约 4GB）
> - **上游** —— [bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)

HiggsTTS（4B）两个库各移植了一份：Python 在 [`example/python/higgstts_vt/`](../example/python/higgstts_vt/)，
C# 在 [`example/CSharp/HiggsTtsSharp/`](../example/CSharp/HiggsTtsSharp/)（命令行在 `HiggsTts.Net/`），都自包含。
参考音频 → RVQ 码 → 塞进 prompt → Qwen3 36 层 AR → DAC 解码。

两份实现**逐位对齐**（DAC 解码和 AR logits 一致），速度 **0.2~0.3 RTF**，耗时几乎全在
36 层 backbone AR。

---

## IndexTTS 2.5

> - **详细文档** —— **[indextts.md](indextts.md)**
> - **权重** —— [NeemaShioSe/IndexTTS2.5.gguf](https://huggingface.co/NeemaShioSe/IndexTTS2.5.gguf)（约 3.3GB）
> - **上游** —— [index-tts/index-tts](https://github.com/index-tts/index-tts)

IndexTTS 2.5（零样本音色克隆 + 情绪控制，中/英/日/西等 99 种语言）的 C# 移植，
整条链路（文本前端 → GPT-2 AR → 语义 codec → s2mel/CFM → BigVGAN，外加参考音频的
fbank/Wav2Vec2-BERT/CAMPPlus）都在
[`example/CSharp/IndexTtsSharp/`](../example/CSharp/IndexTtsSharp/)（命令行在 `IndexTts.Net/`），自包含。

`--emo` 可混 8 种情绪，`--text` 里可以写 `<字|读音>` 发音标注；**RTF 0.314**（f16）/ **0.298**（q8）。

---

## PP-OCRv6（PaddleOCR）

> - **详细文档** —— **[paddleocr.md](paddleocr.md)**
> - **权重** —— [NeemaShioSe/paddleocr.gguf](https://huggingface.co/NeemaShioSe/paddleocr.gguf)（rec + det + dict）
> - **上游** —— [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

PaddleOCR 的 **PP-OCRv6_medium_det / _rec** 移植，只做了 Python 端
（[`example/python/ocr_vt/`](../example/python/ocr_vt/)），自包含 `.pyd`。输入裁好的单行
（或一整块区域），输出文本——整页怎么切行是调用方的事。

识别默认只跑 rec，`det=True` 才先检测再识别。识别结果与 CPU 版一致率 **99.95%**，
比 CPU 流水线快 **7~14×**，固定输入尺寸下和 PyTorch(ROCm) 打平甚至更快一丢丢（0.084 s vs 0.10 s）。

---

## PaddleOCR-VL 1.6

> - **详细文档** —— **[paddleocrvl.md](paddleocrvl.md)**
> - **上游** —— [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

0.3B 的**视觉语言模型**（ERNIE-4.5-0.3B + 改过的 SigLIP 视觉塔），Python 端
（[`example/python/paddleocrvl_vt/`](../example/python/paddleocrvl_vt/)）。喂图 + 任务前缀
（`OCR:` / `Table Recognition:` / …），出结构化文本。

输出与 llama.cpp(ROCm) 的 llama-server **逐字一致**；整张 937 token 的截图 0.77 s
（视觉塔占 0.46 s，解码稳态 482 tok/s）。
