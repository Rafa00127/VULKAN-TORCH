# VULKAN-TORCH

**🌏 Language / 语言:** **中文** · [English](README.en.md)

## 总结和碎碎念

使用了ggml-vulkan的部分作为核心，实现两个类似torch的库（python和c#各一个）。

你可能觉得——“咦，这玩意儿，又不ggml又不pytorch，那不纯区嘛”

你可能还真没错，但换个角度想，它又轻（50mb左右）又快（ggml集美底子好），拿来快速移植小模型还不用像ggml那样写c++，那不又神了嘛。

总之它可以用于移植一些小模型玩玩，因为是vulkan做后端，所以全平台支持，这里提供了两个库，一个python库vulkantorch，一个c#库VulkanTorch.net。

这俩功能差不多，但仓库里用他们移植的小模型不一样。

废话，我不想再重复造了轮子了，token太多没地方花可以喂肥鱼。

在示例中就搞了两个tts模型和一个ocr模型（paddle）的移植示范，可以参考也可以自己拿来玩。

对的对的，这还是重复造了两次轮子用于演示某TTS模型的推理作为演示。


接下来请小肥鱼来讲解一下怎么来玩这两个库吧

---

## 小肥鱼老师的讲解

**vulkan-torch** 是跑在 [ggml](https://github.com/ggml-org/ggml) Vulkan 后端上的迷你张量库，API 像 PyTorch：

- **`vulkantorch`** — Python 库（pybind11）
- **`VulkanTorch.Net`** — C# 库（P/Invoke）

共用同一个 C++ 核（`src/`）和同一个 ggml，行为一致。

| | |
|---|---|
| 定位 | 推理专用，无 autograd |
| 执行模型 | eager 写法，内部把每段前向**捕获成一张 ggml 图**，读到结果才算 |
| 设备 | 显式 `Device` + `Memory`（权重常驻显存） |
| 布局 | 对上层是 PyTorch row-major，内部桥接到 ggml 反向的 `ne[]` |
| 依赖 | 自带 ggml（静态链进产物），运行时只要 Vulkan 驱动 |

每个语言一份 native：`build/vulkantorch.dll`（C#）、`vulkantorch/_vulkantorch*.pyd`（Python），互不依赖——只写 Python 的人不用装 .NET，反之亦然。

---

## 环境要求

| | |
|---|---|
| OS | 不限。C++ 核跨平台；仓库只带了 Windows 的构建脚本 `build_win.py`，Linux/mac 直接 `cmake` 即可 |
| 编译器 | MSVC / GCC 均可 |
| Python | 3.12 |
| Vulkan | 显卡驱动带 Vulkan 运行时 |
| .NET | SDK 10（只在编 C# 示例时需要） |

---

## 构建

```bash
python build_win.py
```

或者用 GCC（MinGW/w64devkit）跑 `build.sh`——同样的东西，只是走 gcc/ninja：

```bash
sh build.sh        # -> build-gcc/，输出 build-gcc/vulkantorch.dll
```

产出：

```
build/vulkantorch.dll                          # 共享库（给 C# P/Invoke）
vulkantorch/_vulkantorch.cp312-win_amd64.pyd   # Python 扩展（直接生在包里）
```

没有 pybind11 时 Python 扩展会跳过，`vulkantorch.dll` 照常产出。

> `build.sh` 走 GCC，默认关掉 Python 模块（`BUILD_PYTHON=OFF`），只出 `vulkantorch.dll`；想连 Python 扩展一起编就加 `-DBUILD_PYTHON=ON -Dpybind11_DIR=...`。

---

## 目录结构

```
src/                          C++ 核
vulkantorch/                  Python 库
VulkanTorch.Net/              C# 库
example/                      各种示例小模型 c#有两个tts，py有一个tts和一个ocr
data/ref_audio/               参考音频 
bench/                        一些测试用的脚本
third_party/ggml/             只剩下cpu和vk的ggml
```

---

## 想写代码？

**if you wanna learn how to write code, check here** → **[sharing_docs/how-to-write-code.md](sharing_docs/how-to-write-code.md)**

覆盖两个库的安装/引用、最小示例、复用同一张图，以及权重该摆成什么形状。

---

## Python：`vulkantorch`

### 跑 TTS 示例（`cli.py`）

[example/python/higgstts_py/cli.py](example/python/higgstts_py/cli.py)：**参考音频 + 文本 → wav**（`encode_ref` + 自回归 + DAC 解码）。

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
(python这边 librosa 初始化浪费了 0.8s —— **只在第一次调用里**：同一个进程里再调一次 enc 只要 41 ms，
总时间落到 1300 ms / RTF 0.23，和 C# 那边基本一致。CLI 每次都是新进程，所以打印出来必然含这 0.8s。)

| 参数 | 说明 |
|---|---|
| `--model` | **必填**，HiggsTTS 的 GGUF；[下载](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf)（约 4GB） |
| `--tokenizer` | 默认用仓库里那份 `data/ref_audio/higgs_tts_v3_tokenizer.json` |
| `--ref-wav` | 参考音频，默认 `data/ref_audio/melinaref_24k.wav` |
| `--ref-text` / `--text` | 参考文本 / 要合成的文本（可加 `<|style:whispering|>` 等风格标签） |
| `--out` | 输出 wav，默认 `data/higgstts/cli_py.wav` |
| `--temperature` `--topk` `--seed` | 采样参数，默认 `0.9 / 50 / 42` |
| `--max-steps` | 自回归步数上限，`0` = 按文本长度自动预测 |
| `--encode-only` / `--no-graph-cache` | 只跑 `encode_ref` / 关掉图缓存（A/B 用） |

---

## C#：`VulkanTorch.Net`

### 跑 TTS 示例（`HiggsTts.Net.exe`）

[example/CSharp/HiggsTts.Net/](example/CSharp/HiggsTts.Net/) 是 `higgstts_py` 的 C# 对等实现，同样自包含（自带分词器 + 重采样器，不需要 Python 侧导出任何东西）。

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

两个模式：`synth`（默认，参考音频+文本→wav）、`encode`（只跑 `encode_ref`）。参数与 Python 版一一对应，另有 `--no-graph-cache`；`-h` / `--help` 打印全部参数。

---

## 三语言速度对比
基准测试在 [bench/](bench/)。

c#和py使用同一个 C++ 核，只有绑定层不同。

省流就是：”它们都差不多“。

c++版higgstts在本项目中就不重复造轮子了，直接用[这个项目](https://github.com/Rafa00127/HiggsTTS.cpp)，都为同一个ggml-vulkan后端，c++版没开图缓存，因为实测对于此情形，图缓存对c++版而言区别不太大，不过用图缓存来对比不开图缓存的c++版还是有一点点小小的不公平。

**单算子**（ms）：

| | C++ | C# | Python |
|---|---|---|---|
| matmul 4096³ | 3.75 | 3.60 | 3.72 |
| conv1d 16384 512→512 | 2.39 | 2.26 | 2.65 |

**长序列自回归**（HiggsTTS AR，约 550 帧，ms/帧，20+s的长音频合成）：

| | C++（参考） | C# | Python |
|---|---|---|---|
| 每步重建图 | 8.12 | 8.83 | 9.42 |
| 图缓存 | — | **7.46** | **7.64** |

**图缓存版和重建版的 wav 是逐位相同的**（不是"差不多"），所以图缓存只省时间、不动数值。

对于上述示例中的长音频合成（约 550 帧 / 21.5s 音频），其RTF：**C++ 0.207 / C# 0.195 / Python 0.240**（Python 含一次 ~0.9s 的 librosa 初始化）。

---

## 示例模型：HiggsTTS

> **权重下载** —— [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf)（约 4GB）

两边的示例都把 [HiggsTTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)（4B TTS）移植了过来：

```
参考音频 ──► encode_ref ──► RVQ 码 ──┐
             (codec)                 ├──► Qwen3 36层 AR ──► 码 ──► DAC 解码 ──► wav
   文本 ────► tokenizer ────► prompt ┘       (backbone)              (decoder)
```

- **Python**：`example/python/higgstts_py/`（`model.py` / `ar.py` / `decode.py` / `tts.py`）
- **C#**：`example/CSharp/HiggsTts.Net/`（`EncodeRef.cs` / `Ar.cs` / `DacDecoder.cs` / `HiggsTokenizer.cs` / `Resampler.cs`）
- **权重**：[NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf)（约 4GB，仓库不含）

两者数值上对齐：DAC 解码和 AR 的 logits **逐位一致**，tokenizer 输出也一致；`encode_ref` 用同一个重采样器时 97% 的帧相同（差异只来自 Python 用 librosa、C# 用自带 Kaiser 重采样，听感无差别）。速度约 **0.2~0.3 RTF**。

---

## 示例模型：IndexTTS 2.5

> **权重下载** —— [NeemaShioSe/IndexTTS2.5.gguf](https://huggingface.co/NeemaShioSe/IndexTTS2.5.gguf)（约 3.3GB）

只做了 C# 端（懒）。零样本音色克隆 + 情绪控制，中/英/日/西等 99 种语言。
整条链路（文本前端 → GPT-2 AR → 语义 codec → s2mel/CFM → BigVGAN，外加参考音频的
fbank/Wav2Vec2-BERT/CAMPPlus）都在 [example/CSharp/IndexTts.Net/](example/CSharp/IndexTts.Net/) 里，自包含。

```
参考音频 ─┬─ Kaldi fbank ─► CAMPPlus ──────────────────────► style
          ├─ Kaldi fbank ─► Wav2Vec2-BERT ─► spk_cond ─┐
          └─ 22.05k mel ───────────────────────────────┤
                                                       ▼
文本 ─► 分词/归一化/分段 ─► GPT-2 AR ─► codec ─► length_regulator ─► CFM(25 步 Euler+CFG) ─► BigVGAN ─► wav
```

```bat
dotnet build example\CSharp\IndexTts.Net -c Release

example\CSharp\IndexTts.Net\bin\Release\net10.0\IndexTts.Net.exe synth ^
  --model model\indextts2.5\indextts2.5.f16.gguf ^
  --ref-wav data\ref_audio\melinaref_24k.wav ^
  --text "大家好，这是一个测试。" --out data\indextts\out.wav
```

| 参数 | 说明 |
|---|---|
| `--model` | **必填**，IndexTTS 2.5 的 GGUF；[下载](https://huggingface.co/NeemaShioSe/IndexTTS2.5.gguf)（约 3.3GB，或用 `tools/convert/convert_index_tts2_to_gguf.py` 自己转） |
| `--ref-wav` | 参考音频（音色来源），默认 `data/ref_audio/melinaref_24k.wav` |
| `--text` | 要合成的文本，可加 `<字\|读音>` 发音标注 |
| `--emo` | 情绪权重，如 `--emo "happy=0.6,calm=0.4"`（8 种：happy/angry/sad/afraid/disgusted/melancholic/surprised/calm） |
| `--lang` `--out` `--seed` `--max-steps` | 语言 / 输出 / 采样种子 / 步数上限 |
| `--top-p` `--top-k` `--temperature` `--rep-penalty` | 采样参数（默认 0.8 / 30 / 0.8 / 10） |
| `--num-beams` | 束搜索宽度，默认 1 |
| `--cfg-rate` `--diffusion-steps` | CFM 的 CFG 强度和扩散步数（默认 0.7 / 25） |
| `--duration-factor` | 语速/时长缩放 |
| `--no-graph-cache` | 关掉 AR 的图缓存（A/B 用） |
| `-h` `--help` | 打印全部参数 |

### 速度（RX 7900 XTX / Vulkan / f16 GGUF）

`synth` 单句合成，各阶段耗时与 RTF（RTF = 推理耗时 / 音频时长，和官方一样不含权重加载）：

```bat
IndexTts.Net.exe synth --model model\indextts2.5\indextts2.5.f16.gguf ^
  --ref-wav data\ref_audio\melinaref_24k.wav --text "..." --out data\indextts\out.wav
```

| 文本长度 | 音频 | GPT-2 AR | s2mel (codec+CFM) | BigVGAN | 推理总 | RTF |
|---|---|---|---|---|---|---|
| 11 字 | 3.51 s | 625 ms | 631 ms | 475 ms | 1.98 s | 0.564 |
| 33 字 | 7.30 s | 1059 ms | 797 ms | 852 ms | 2.95 s | 0.404 |
| 89 字 | 19.04 s | 2384 ms | 2045 ms | 2220 ms | 6.90 s | 0.362 |

对照官方在 **RTX 4090** 上公布的 2.5 RTF（`kv_cache=True`）：bf16 ~0.20、fp32 ~0.21（7~200 字）。
本移植折算约慢 **1.75×**，差距主要来自硬件（7900 XTX vs 4090）和官方用 bf16 + CUDA 融合算子，这里是 f16 GGUF + 通用 Vulkan 后端。
长句单段时 AR / s2mel / BigVGAN 三段耗时几乎均分。权重加载约 2–3 s（不计入上表）。

## 示例模型：PP-OCRv6（PaddleOCR）

> **权重下载** —— [NeemaShioSe/paddleocr.gguf](https://huggingface.co/NeemaShioSe/paddleocr.gguf)（rec + det + dict）

文本检测 + 识别，只做了 Python 端（`example/python/ocr_py/`），自包含 `.pyd`。
用法、精度/速度结论见 [example/python/PaddleOCR.md](example/python/PaddleOCR.md)。

## 两个 TTS 移植的横向对比

同一台机（RX 7900 XTX / Vulkan）、同一句中文（85 个汉字）、同一个参考音频、`--seed 42`，
各跑 5 次取最好（丢掉第一次）。RTF 不含权重加载：

| 模型 | 音频 | 推理耗时 | RTF |
|---|---|---|---|
| HiggsTTS v3（~4B，`higgs-v3-tts.gguf`，9.34 GB） | 23.76 s | 6.86 s | **0.288** |
| IndexTTS 2.5（~0.8B，`indextts2.5.f16.gguf`，3.3 GB） | 18.40 s | 5.78 s | **0.314** |
| IndexTTS 2.5（量化版，`indextts2.5.q8.gguf`，2.0 GB） | 17.76 s | 5.29 s | **0.298** |

IndexTTS 现在单句绝对耗时**反超** Higgs（5.78 vs 6.86 s）。Higgs 生成的音频更长（23.76 vs 18.40 s），
所以它的 RTF 更低 —— RTF 受"一句话生出多长音频"影响很大，别只看它。
Higgs 的 `Decode` 几乎免费（41 ms，耗时全在 36 层 backbone AR），IndexTTS 则是 AR / s2mel / BigVGAN 三家平分。
测试句和完整命令见 [IndexTTS2.5.md](example/CSharp/IndexTTS2.5.md)。

## 许可

MIT
