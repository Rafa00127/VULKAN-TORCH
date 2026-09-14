# VULKAN-TORCH

## 总结和碎碎念

使用了ggml-vulkan的部分作为核心，实现两个类似torch的库（python和c#各一个）。

你可能觉得——“咦，这玩意儿，又不ggml又不pytorch，那不纯区嘛”

你可能还真没错，但换个角度想，它又轻（50mb左右）又快（ggml集美底子好），拿来快速移植小模型还不用像ggml那样写c++，那不又神了嘛。

总之它可以用于移植一些小模型玩玩，因为是vulkan做后端，所以全平台支持，这里提供了两个库，一个python库vulkantorch，一个c#库VulkanTorch.net。

这俩功能差不多，但仓库里用他们移植的小模型不一样。

废话，我不想再重复造了论轮子了，token太多没地方花可以喂肥鱼。

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

产出：

```
build/vulkantorch.dll                          # 共享库（给 C# P/Invoke）
vulkantorch/_vulkantorch.cp312-win_amd64.pyd   # Python 扩展（直接生在包里）
```

没有 pybind11 时 Python 扩展会跳过，`vulkantorch.dll` 照常产出。

---

## 目录结构

```
src/                          C++ 核（runtime / graph / tensor / memory / ops / gguf + C ABI）
vulkantorch/                  Python 库（__init__.py + pybind 源码 + 编译出的 .pyd）
VulkanTorch.Net/              C# 库（P/Invoke 封装）
example/python/higgstts_py/   Python 示例：HiggsTTS 移植 + CLI
example/CSharp/HiggsTts.Net/  C# 示例：HiggsTTS 移植 + CLI
example/CSharp/IndexTts.Net/  C# 示例：IndexTTS 2.5 移植 + CLI（零样本克隆 + 情绪控制）
example/CSharp/VulkanTorch.Test/   冒烟测试（matmul / conv1d）
example/CSharp/VulkanTorch.Bench/  三语言基准之一
data/ref_audio/               参考音频 + tokenizer（已入库）
bench/                        三语言基准
third_party/ggml/             自带 ggml（只留 CPU + Vulkan）
```

---

## Python：`vulkantorch`

### 1. 安装

没发 PyPI。跑完 `python build_win.py`，把**仓库根**加进 `sys.path` 即可：

```python
import sys
sys.path.insert(0, r"path/to/vulkan-torch")
import vulkantorch as vt
```

扩展自带 ggml，不用配任何 DLL 目录。

### 2. 写代码

一个卷积 + 一个线性层：

```python
import numpy as np
import vulkantorch as vt

rt = vt.Runtime()
gpu = rt.gpu()

# 权重放显存：Memory 是显存里的存储，往里塞张量
wc = np.random.rand(64, 32, 3).astype(np.float32)      # conv   [Cout, Cin, K]
wl = np.random.rand(128, 64).astype(np.float32)        # linear [out, in]
mwc = vt.Memory(gpu, wc.nbytes).tensor([64, 32, 3], wc.tobytes())
mwl = vt.Memory(gpu, wl.nbytes).tensor([128, 64], wl.tobytes())

# 一段前向 = 一个 Graph
g = vt.Graph(rt, gpu)
g.enter()
x = g.input([64, 32], np.zeros((64, 32), np.float32).tobytes())   # [T, Cin]
h = vt.gelu(vt.conv1d(x, mwc, 1, 1, 1))                          # [T, 64]
y = vt.linear(h, mwl)                                            # [T, 128]
y.mark_output()                                                  # 要读的必须标成输出
out = np.frombuffer(y.to_bytes(), np.float32).reshape(64, 128)    # 读 = 计算
g.exit()
```

要点：

- `enter()` / `exit()` 之间只是**记账**，`to_bytes()` / `to_host()` 才真正计算。
- 要读的张量必须 `mark_output()`，否则显存会被调度器复用。
- `Memory` 要在 `enter()` **之前**建好；算子跑在**权重的后端**上（权重在显存，算子就在 GPU）。
- 算子都是 `vt.` 下的函数：`matmul` `linear` `conv1d` `rms_norm` `layer_norm` `gelu` `silu` `soft_max` `rope` `flash_attn` `snake_1d` `im2col_rafa` …
- 输入用 `g.input(shape, bytes)`（F32）/ `g.input_i32(shape, bytes)`。

从 GGUF 读权重：

```python
f = vt.GgufFile(r"model.gguf", gpu)
w = f.tensor("some.weight")     # 拷进显存，保留原量化类型
```

> float32 matmul 走 Vulkan 的 fp16 矩阵核，和双精度参考差 ~1e-3，正常。这是**设备级**默认，所有模型共用；想全程走 f32，在 `import vulkantorch` **之前**设环境变量 `GGML_VK_DISABLE_F16=1`（C# 在首次调 `vt_*` 前 `SetEnvironmentVariable`）。启动日志那行 `fp16:` 就是当前状态。

### 3. 复用同一张图

形状不变、只有数据变的话，图可以**分配一次**、之后只换输入重算——循环里省掉重复建图和重复分配：

```python
g = vt.Graph(rt, gpu)
g.enter()
x = g.input([64, 32], np.zeros((64, 32), np.float32).tobytes())
y = vt.linear(vt.gelu(vt.conv1d(x, mwc, 1, 1, 1)), mwl)
y.mark_output()
g.exit()

g.alloc_static()                             # 建完之后调一次
for i in range(10):
    g.set_input(x, batch(i))                 # 换输入
    g.compute_static()                       # 只算，不再分配
    out = np.frombuffer(y.to_bytes(), np.float32).reshape(64, 128)
```

要点：

- **`compute()` 是一次性的**，只能调一次；重放必须用 `alloc_static()` + `compute_static()`。
  （调度器每次重新分配会让结果取决于分配那一遍、而不是输入——这不是性能差异，是**算错**。）
- 被重放的图必须**位置无关**：位置`用 set_rows` 当运行时输入、注意力窗口钉在固定尺寸、mask 也当输入。参考 [GraphCache.cs](VulkanTorch.Net/GraphCache.cs) 里 bucket 的做法。
- 一般小模型无所谓；但**自回归**那种每生成一个 token 就要跑一遍的场合，省下来的就是实打实的（见 [速度对比](#三语言速度对比)）。

### 4. 跑 TTS 示例（`cli.py`）

[example/python/higgstts_py/cli.py](example/python/higgstts_py/cli.py)：**参考音频 + 文本 → wav**（`encode_ref` + 自回归 + DAC 解码）。

```bat
python example\python\higgstts_py\cli.py ^
  --model path/to/HiggsTTS3.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --ref-wav data/ref_audio/melinaref_24k.wav ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

```
=== Timing ===
Prefill:           933 ms
Backbone AR:      1261 ms
Decode:             10 ms
Total:            2204 ms
Audio:            5.72 sec
RTF:             0.385 x
```
(python这边librosa初始化浪费了0.8s)

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

### 1. 引用

没发 NuGet。项目引用（或引 `VulkanTorch.Net.dll`），再把**原生库放到 exe 旁边**：

```xml
<ItemGroup>
  <ProjectReference Include="..\VulkanTorch.Net\VulkanTorch.Net.csproj" />
  <None Include="..\build\vulkantorch.dll" Link="vulkantorch.dll" CopyToOutputDirectory="PreserveNewest" />
</ItemGroup>
```

（ggml 已静态链进去，就这一个文件。）

### 2. 写代码

同样是一个卷积 + 一个线性层：

```csharp
using VulkanTorch;

using var rt = new Runtime();
var gpu = rt.Gpu();

int T = 64, Cin = 32, Cout = 64, Lout = 128, K = 3;
var wc = new float[Cout * Cin * K];      // 填上你的数据（row-major）
var wl = new float[Lout * Cout];

using var mwc = new Memory(gpu, (ulong)wc.Length * 4);   // 权重放显存
using var mwl = new Memory(gpu, (ulong)wl.Length * 4);
var twc = mwc.Tensor(new long[] { Cout, Cin, K }, Ops.F32, ToBytes(wc));
var twl = mwl.Tensor(new long[] { Lout, Cout }, Ops.F32, ToBytes(wl));

using var g = new Graph(rt, gpu);
g.Enter();
var x = g.Input(new long[] { T, Cin }, new float[T * Cin]);
var y = Ops.Linear(Ops.Gelu(Ops.Conv1d(x, twc, 1, 1, 1)), twl).MarkOutput();
var outv = y.ToFloats(g);                   // 读 = 计算
g.Exit();

static byte[] ToBytes(float[] f)
{
    var b = new byte[f.Length * 4];
    Buffer.BlockCopy(f, 0, b, 0, b.Length);
    return b;
}
```

要点与 Python 一致：`Enter()`/`Exit()` 是捕获作用域，`ToFloats`/`ToBytes`/`ToInts` 触发计算；要读的张量 `MarkOutput()`；`Memory` 在 `Enter()` 之前建好；算子在 `Ops` 静态类里。

### 3. 复用同一张图

```csharp
using var g = new Graph(rt, gpu);
g.Enter();
var x = g.Input(new long[] { T, Cin }, new float[T * Cin]);
var y = Ops.Linear(Ops.Gelu(Ops.Conv1d(x, twc, 1, 1, 1)), twl).MarkOutput();
g.Exit();

g.AllocStatic();                            // 建完之后调一次
for (int i = 0; i < 10; i++)
{
    g.SetInput(x, batch(i));                // 换输入
    g.ComputeStatic();                      // 只算，不再分配
    var outv = y.ToFloats(g);
}
```

与 Python 版完全一致：`Compute()` 一次性的，重放用 `AllocStatic()` + `ComputeStatic()`；被重放的图必须位置无关。自回归解码建议直接用 [GraphCache.cs](VulkanTorch.Net/GraphCache.cs) 的 `BucketedReplay`（按窗口 bucket 建一次图、自动分配与重放），模型侧只需提供"怎么建图"和"每帧输入是什么"两个回调。

### 4. 跑 TTS 示例（`HiggsTts.Net.exe`）

[example/CSharp/HiggsTts.Net/](example/CSharp/HiggsTts.Net/) 是 `higgstts_py` 的 C# 对等实现，同样自包含（自带分词器 + 重采样器，不需要 Python 侧导出任何东西）。

```bat
dotnet build example\CSharp\HiggsTts.Net -c Release

example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe synth ^
  --model D:\models\HiggsTTS3.gguf ^
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

单算子三语言没差别（GPU 派发主导）；自回归每步要重建 ~400 算子的图，C#/Python 穿过绑定层调这 400 次，开销才显出来——用上 [§3 的图缓存](#3-复用同一张图) 就消掉了，两边追平/略超过原生 C++。

**图缓存版和重建版的 wav 是逐位相同的**（不是"差不多"），所以图缓存只省时间、不动数值。

对于上述示例中的长音频合成（约 550 帧 / 21.5s 音频），其RTF：**C++ 0.207 / C# 0.195 / Python 0.240**（Python 含一次 ~0.9s 的 librosa 初始化）。

---

## 示例模型：HiggsTTS

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

只做了 C# 端（成果要直接给 WPF 听书应用用）。零样本音色克隆 + 情绪控制，中/英/日/西等 99 种语言。
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
| `--model` | **必填**，IndexTTS 2.5 的 GGUF（4 个权重文件用 `tools/convert/convert_index_tts2_to_gguf.py` 转，约 3.3GB） |
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
| 11 字 | 3.51 s | 473 ms | 634 ms | 496 ms | 1.95 s | 0.557 |
| 33 字 | 8.30 s | 1003 ms | 958 ms | 1073 ms | 3.39 s | 0.408 |
| 89 字 | 17.89 s | 2107 ms | 1885 ms | 2107 ms | 6.45 s | 0.361 |

对照官方在 **RTX 4090** 上公布的 2.5 RTF（`kv_cache=True`）：bf16 ~0.20、fp32 ~0.21（7~200 字）。
本移植折算约慢 **1.7×**，差距主要来自硬件（7900 XTX vs 4090）和官方用 bf16 + CUDA 融合算子，这里是 f16 GGUF + 通用 Vulkan 后端。
长句单段时 AR / s2mel / BigVGAN 三段耗时几乎均分。权重加载约 2 s（不计入上表）。

## 两个 TTS 移植的横向对比

同一台机（RX 7900 XTX / Vulkan）、同一句 89 字中文，各跑一次（RTF 不含权重加载）：

| 模型 | 音频 | 推理耗时 | RTF |
|---|---|---|---|
| HiggsTTS v3（~4B，`higgs-v3-tts.gguf`，9.34 GB） | 23.72 s | 6.87 s | **0.29** |
| IndexTTS 2.5（~0.8B，`indextts2.5.f16.gguf`，3.3 GB） | 17.89 s | 6.45 s | **0.361** |

两者单句绝对耗时接近（~6.7–6.9 s）；Higgs 这次生成的音频更长（23.72 vs 17.89 s），所以 RTF 更低。
Higgs 的 `Decode` 几乎免费（43 ms，耗时全在 36 层 backbone AR），IndexTTS 则是 AR / s2mel / BigVGAN 三家平分。

## 许可

MIT
