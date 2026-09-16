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

## 示例模型

三个小模型（两个 TTS + 一个 OCR）的移植，代码在 `example/`，文档在 `sharing_docs/`：

→ **[sharing_docs/example-models.md](sharing_docs/example-models.md)** —— 总览（简介、速度、权重下载）

| 模型 | 语言 | 详细 |
|---|---|---|
| HiggsTTS v3 | C# + Python | [sharing_docs/higgstts.md](sharing_docs/higgstts.md) |
| IndexTTS 2.5 | C# | [sharing_docs/indextts.md](sharing_docs/indextts.md) |
| PP-OCRv6 | Python | [sharing_docs/paddleocr.md](sharing_docs/paddleocr.md) |

---

## 许可

MIT
