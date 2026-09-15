# 怎么写代码

**🌏 Language / 语言:** **中文** · [English](how-to-write-code.en.md)

[README](../README.md) 的配套文档：环境要求和构建在那份里，这里只讲怎么写代码。

---

## 共同约定：权重怎么摆（`Linear`）

线性层的权重必须按 **`(out, in)`** 摆：

| | 写法 | 权重形状（前者为行） |
|---|---|---|
| 数学课 | `X @ W + b` | `W (in, out)` —— 输入维在前 |
| PyTorch `nn.Linear` | `x @ W.T + b` | `W (out, in)` —— 输出维在前 |
| 本库 | `Linear(x, w)` = `ggml_mul_mat(w, x)` = `x @ w.T` | `w (out, in)` —— 同 PyTorch |

**三者记法不同，但内存布局是一致的**：torch 的行优先和 ggml 的内维最快正好互逆，而 ggml 要的
收缩维 `in` 落在 `ne0` 也正好是这个朝向 —— 所以按 torch 的样子原样读进去就是对的，不用转置。

- 权重按 `(out, in)` 存，用 `Linear(x, w)`
- **w 为权重时，别写 `Matmul(x, w)`**：它得先 `cont(transpose(w))` 去凑 ggml 的约定 ——
  f16 白搬带宽，量化权重会爆炸。`Matmul` 留给激活 × 激活
- 上游若是 HF `Conv1D`（名字骗人：它就是个线性层、没有卷积核，只是权重存 `(in, out)`），
  在 converter 里转一次再写入

---

## Python：`vulkantorch`

### 1. 安装

没发 PyPI。跑完 `python build_win.py`或者`build.sh`，把**仓库根**加进 `sys.path` 即可：

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
- **`shape()` 会吃掉最慢维的 1**（至少留一维）：

  | PT 形状 | `shape()` |
  |---|---|
  | `(2, 1, 4)` | `(2, 1, 4)` |
  | `(1, 4)` | `(4,)` |
  | `(1, 2, 4)` | `(2, 4)` |

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
- 被重放的图必须**位置无关**：位置`用 set_rows` 当运行时输入、注意力窗口钉在固定尺寸、mask 也当输入。参考 [GraphCache.cs](../VulkanTorch.Net/GraphCache.cs) 里 bucket 的做法。
- 一般小模型无所谓；但**自回归**那种每生成一个 token 就要跑一遍的场合，省下来的就是实打实的（见 [速度对比](../README.md#三语言速度对比)）。

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

与 Python 版完全一致：`Compute()` 一次性的，重放用 `AllocStatic()` + `ComputeStatic()`；被重放的图必须位置无关。自回归解码建议直接用 [GraphCache.cs](../VulkanTorch.Net/GraphCache.cs) 的 `BucketedReplay`（按窗口 bucket 建一次图、自动分配与重放），模型侧只需提供"怎么建图"和"每帧输入是什么"两个回调。
