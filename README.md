# VULKAN-TORCH

## 总结和碎碎念

总之项目使用了ggml-vulkan的部分，做了两个类似torch的库（python和c#各一个），可以用于移植一些小模型玩玩，因为是vulkan做后端，所以全平台支持，在示例中就搞了两个tts小模型的移植示范，可以参考也可以自己拿来玩。

对的对的，这算是重复造了两次轮子。

接下来请小肥鱼来讲解一下怎么使用这两个库吧

---

## 小肥鱼老师的讲解

**vulkan-torch** 是跑在 [ggml](https://github.com/ggml-org/ggml) Vulkan 后端上的迷你张量库，API 长得像 PyTorch：

- **`vulkantorch`** — Python 库（pybind11）
- **`VulkanTorch.Net`** — C# 库（P/Invoke）

两者共用同一个 C++ 核（`src/`）和同一个 ggml，行为一致。

| | |
|---|---|
| 定位 | **推理专用**，无 autograd |
| 执行模型 | eager 写法，内部**每段前向捕获成一张 ggml 图**，读到结果才算 |
| 设备 | 显式 `Device` + `Memory`（权重常驻显存） |
| 布局 | 对上层是 PyTorch 的 row-major，内部桥接到 ggml 反向的 `ne[]` |
| 依赖 | 自带 ggml（静态链进产物），运行时只要 Vulkan 驱动 |

每个语言一份 native：`build/vulkantorch.dll`（给 C#）、`vulkantorch/_vulkantorch*.pyd`（给 Python），各自把 ggml 静态吞进去，互不依赖——只写 Python 的人不用装 .NET，反之亦然。

---

## 环境要求

| | |
|---|---|
| OS | 不限。C++ 核跨平台；仓库只带了 Windows 的构建脚本 `build_win.py`（MSVC + VS 自带 Ninja/CMake），Linux/mac 直接 `cmake` 即可 |
| 编译器 | MSVC / GCC 均可 |
| Python | 3.12（扩展按 `cp312` 编） |
| Vulkan | 显卡驱动带 Vulkan 运行时 |
| .NET | SDK 10（只在编 C# 示例时需要） |

---

## 构建

```bash
python build_win.py
```

产出：

```
build/vulkantorch.dll                          # C 用的共享库（供 C# P/Invoke）
vulkantorch/_vulkantorch.cp312-win_amd64.pyd   # Python 扩展（直接生在包里）
```

Python 扩展是**可选**的：没有 pybind11 就跳过，`vulkantorch.dll` 照常产出。

---

## 目录结构

```
src/                          C++ 核（runtime / graph / tensor / memory / ops / gguf + C ABI）
vulkantorch/                  Python 库（__init__.py + pybind 源码 + 编译出的 .pyd）
VulkanTorch.Net/              C# 库（P/Invoke 封装）
example/python/higgstts_py/   Python 示例：HiggsTTS 移植 + CLI
example/CSharp/HiggsTts.Net/  C# 示例：HiggsTTS 移植 + CLI
example/CSharp/VulkanTorch.Test/   冒烟测试（matmul / conv1d）
example/CSharp/VulkanTorch.Bench/  三语言基准之一
data/ref_audio/               参考音频 + tokenizer（已入库）
bench/                        三语言基准
third_party/ggml/             自带 ggml（只留 CPU + Vulkan）
```

---

## Python：`vulkantorch`

### 1. 安装

没发 PyPI，源码构建。跑完 `python build_win.py`，把**仓库根**加进 `sys.path` 即可：

```python
import sys
sys.path.insert(0, r"path/to/vulkan-torch")
import vulkantorch as vt
```

或设一次环境变量：`set PYTHONPATH=path\to\vulkan-torch`

扩展自带 ggml，不用配任何 DLL 目录。

### 2. 写代码跑模型

```python
import sys
sys.path.insert(0, r"path/to/vulkan-torch")
import numpy as np
import vulkantorch as vt

rt = vt.Runtime()
gpu = rt.gpu()
print(rt.name())                    # Vulkan0

a = np.random.rand(512, 512).astype(np.float32)
b = np.random.rand(512, 512).astype(np.float32)
ta = vt.Memory(gpu, a.nbytes).tensor([512, 512], a.tobytes())   # 权重放显存
tb = vt.Memory(gpu, b.nbytes).tensor([512, 512], b.tobytes())

g = vt.Graph(rt, gpu)
g.enter()
c = vt.matmul(ta, tb)
c.mark_output()                     # 要读的张量必须标成输出
out = np.frombuffer(c.to_bytes(), dtype=np.float32).reshape(512, 512)
g.exit()
```

要点：

- `Graph` 是**捕获作用域**，不是逐算子执行；`to_bytes()` / `to_host()` 才触发计算。
- 要读的张量必须 `mark_output()`，否则显存会被调度器复用。
- `Memory` 不能在捕获期间创建——权重先建好再进 `enter()`。
- 算子跑在**权重的后端**上：权重在 GPU 的 `Memory` 里，算子就在 GPU。
- 输入用 `g.input(shape, bytes)` / `g.input_i32(shape, bytes)`。
- 算子都是 `vt.` 下的函数：`matmul` `linear` `conv1d` `rms_norm` `layer_norm` `gelu` `silu` `soft_max` `rope` `flash_attn` `snake_1d` `im2col_rafa` …

从 GGUF 读权重：

```python
f = vt.GgufFile(r"model.gguf", gpu)
print(f.names()[:5])                # 所有张量名
w = f.tensor("some.weight")         # 拷进显存，保留原量化类型
```

> float32 matmul 走 Vulkan 的 fp16 矩阵核，和双精度参考差 ~1e-3，正常。

### 3. 跑 TTS 示例（`cli.py`）

[example/python/higgstts_py/cli.py](example/python/higgstts_py/cli.py)：**参考音频 + 文本 → wav**（`encode_ref` + 自回归 + DAC 解码）。

```bat
python example\python\higgstts_py\cli.py ^
  --model D:\models\HiggsTTS3.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

输出：

```
Prefill: 379 frames x 8 codebooks (928 ms)
Backbone AR: 150 raw frames (1446 ms)
Decode: 137280 PCM samples (5.72 sec) (10 ms)

=== Timing ===
Prefill:           928 ms
Backbone AR:      1446 ms
Decode:             10 ms
Total:            2383 ms
Audio:            5.72 sec
RTF:             0.417 x

Saved: data/higgstts/cli_py.wav
```

（帧数随采样 RNG 变，不是固定的。）

| 参数 | 说明 |
|---|---|
| `--model` | **必填**，HiggsTTS 的 GGUF；[下载](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf)（约 4GB） |
| `--tokenizer` | 默认用仓库里那份 `data/ref_audio/higgs_tts_v3_tokenizer.json` |
| `--ref-wav` | 参考音频，默认 `data/ref_audio/melinaref_24k.wav` |
| `--ref-text` | 参考音频对应的文本 |
| `--text` | 要合成的文本（可加 `<|style:whispering|>` 等风格标签；`--text ""` 只克隆音色） |
| `--out` | 输出 wav，默认 `data/higgstts/cli_py.wav` |
| `--temperature` `--topk` `--seed` | 采样参数，默认 `0.9 / 50 / 42` |
| `--max-steps` | 自回归步数上限，`0` = 按文本长度自动预测（默认） |
| `--encode-only` | 只跑 `encode_ref`，不合成 |

---

## C#：`VulkanTorch.Net`

### 1. 引用

没发 NuGet，两种办法：

**(a) 项目引用**（推荐）：

```xml
<ItemGroup>
  <ProjectReference Include="..\VulkanTorch.Net\VulkanTorch.Net.csproj" />
</ItemGroup>
```

**(b) 引 DLL**：加 `VulkanTorch.Net.dll`。

两种都要把**原生库放到 exe 旁边**：

```xml
<ItemGroup>
  <None Include="..\build\vulkantorch.dll" Link="vulkantorch.dll" CopyToOutputDirectory="PreserveNewest" />
</ItemGroup>
```

（ggml 已静态链进去，就这一个文件。）

### 2. 写代码跑模型

```csharp
using VulkanTorch;

using var rt = new Runtime();
var gpu = rt.Gpu();
Console.WriteLine(rt.Name);                 // Vulkan0

int M = 512, K = 512, N = 512;
float[] a = new float[M * K], b = new float[K * N];   // 填上你的数据（row-major）

using var ma = new Memory(gpu, (ulong)a.Length * 4);  // 权重放显存
using var mb = new Memory(gpu, (ulong)b.Length * 4);
var ta = ma.Tensor(new long[] { M, K }, Ops.F32, ToBytes(a));
var tb = mb.Tensor(new long[] { K, N }, Ops.F32, ToBytes(b));

using var g = new Graph(rt, gpu);
g.Enter();
var c = Ops.Contiguous(Ops.Matmul(ta, tb)).MarkOutput();
var got = c.ToFloats(g);                    // 读结果，这里才算
g.Exit();

static byte[] ToBytes(float[] f)
{
    var b = new byte[f.Length * 4];
    Buffer.BlockCopy(f, 0, b, 0, b.Length);
    return b;
}
```

要点与 Python 一致：

- `Enter()` / `Exit()` 是捕获作用域；`ToFloats` / `ToBytes` / `ToInts` 触发计算。
- 要读的张量 `MarkOutput()`；`Memory` 在 `Enter()` 之前建好。
- 算子在 `Ops` 静态类（`Ops.Matmul` `Ops.Conv1d` `Ops.RmsNorm` `Ops.FlashAttn` …），形状/元数据在 `Tensor` 上。
- 输入用 `g.Input(shape, float[])` 或 `g.Input(shape, byte[])`。

### 3. 跑 TTS 示例（`HiggsTts.Net.exe`）

[example/CSharp/HiggsTts.Net/](example/CSharp/HiggsTts.Net/) 是 `higgstts_py` 的 C# 对等实现，同样自包含——自带分词器（`HiggsTokenizer`）和重采样器（`Resampler`），不需要 Python 侧导出任何东西。

```bat
dotnet build example\CSharp\HiggsTts.Net -c Release

example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe synth ^
  --model D:\models\HiggsTTS3.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

输出（格式与参考 C++ 的 `higgs_cli.exe` 一致）：

```
Prefill: 379 frames x 8 codebooks (117 ms)
Backbone AR: 400 raw frames (3482 ms)
Decode: 377280 PCM samples (15.72 sec) (25 ms)

=== Timing ===
Prefill:           117 ms
Backbone AR:      3482 ms
Decode:              25 ms
Total:            3625 ms
Audio:           15.72 sec
RTF:             0.231 x

Saved: data\higgstts\cli_cs_synth.wav
```

两个模式：

| 模式 | 作用 |
|---|---|
| `synth`（默认） | 参考音频 + 文本 → wav |
| `encode` | 只跑 `encode_ref`，参考音频 → RVQ 码流 |

参数与 Python 版一一对应：`--model` `--tokenizer` `--ref-wav` `--ref-text` `--text` `--out` `--temperature` `--topk` `--seed` `--max-steps`。

冒烟测试 / 基准：

```bat
example\CSharp\VulkanTorch.Test\bin\Release\net10.0\VulkanTorch.Test.exe     :: matmul / conv1d 对齐
example\CSharp\VulkanTorch.Bench\bin\Release\net10.0\VulkanTorch.Bench.exe   :: 单算子基准
```

---

## 三语言速度对比

同一个 C++ 核、同样的活，只有绑定层不同。基准在 [bench/](bench/)。

**单算子**（权重常驻显存，取 N 次最好）：

| 算子 | C++ | C# | Python |
|---|---|---|---|
| matmul 4096³ | 3.48 ms | 3.41 ms | 3.49 ms |
| conv1d 16384 512→512 K7 | 2.30 ms | 2.37 ms | 2.37 ms |

没有可测差异——算子是毫秒级、GPU 派发主导，绑定开销被噪声淹没。

**长序列自回归**（HiggsTTS AR，约 600 帧）：

| | C++（参考） | C# | Python |
|---|---|---|---|
| 每帧 | **7.97 ms** | 8.64 ms | 9.27 ms |

三个例子都没有使用图缓存，C# 和 Python 的建图开销相比 C++ 多了些，因此有这些差距。

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

两者数值上对齐：

- DAC 解码、AR 的 logits —— **逐位一致**
- tokenizer 输出 —— 完全一致
- `encode_ref` —— 用同一个重采样器时 **97% 的帧完全相同**；差异只来自 Python 用 librosa、C# 用自带 Kaiser 重采样（刻意的：各用各的生态，听感无差别）

速度约 **0.2~0.27 RTF**（好几倍实时）。

---

## 许可

（待定）
