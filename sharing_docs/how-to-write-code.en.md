# How to Write Code

**🌏 Language / 语言:** [中文](how-to-write-code.md) · **English**

Companion to the [README](../README.en.md): environment and build steps live there, this one only covers writing code.

---

## Shared convention: which way the weights go (`Linear`)

A linear layer's weight must be laid out `(out, in)`:

| | Form | Weight shape (first entry is the row) |
|---|---|---|
| Textbook | `X @ W + b` | `W (in, out)` — input dim first |
| PyTorch `nn.Linear` | `x @ W.T + b` | `W (out, in)` — output dim first |
| This library | `Linear(x, w)` = `ggml_mul_mat(w, x)` = `x @ w.T` | `w (out, in)` — same as PyTorch |

**Different notation, identical memory layout**: torch's row-major and ggml's fastest-dim-first are
exact inverses, and ggml wants the contraction dim `in` at `ne0` — which is exactly where an
`(out, in)` weight already lands when read in as-is. No transpose or shuffle needed.

- Store weights as `(out, in)` and call `Linear(x, w)`
- **With a weight as `w`, don't write `Matmul(x, w)`** — it has to `cont(transpose(w))` to satisfy
  ggml's rule: wasted bandwidth on f16, a hard crash on quantized weights. `Matmul` is for activation × activation
- If upstream is an HF `Conv1D` (misleading name: it is a plain linear layer with no kernel, it just
  stores `(in, out)`), transpose it once in the converter

---

## Python: `vulkantorch`

### 1. Install

Not on PyPI. After `python build_win.py` (or `build.sh`), just add the **repo root** to `sys.path`:

```python
import sys
sys.path.insert(0, r"path/to/vulkan-torch")
import vulkantorch as vt
```

The extension bundles ggml — no DLL dir to configure.

### 2. Write Code

One conv + one linear:

```python
import numpy as np
import vulkantorch as vt

rt = vt.Runtime()
gpu = rt.gpu()

# weights on the GPU: Memory is VRAM storage; stuff tensors into it
wc = np.random.rand(64, 32, 3).astype(np.float32)      # conv   [Cout, Cin, K]
wl = np.random.rand(128, 64).astype(np.float32)        # linear [out, in]
mwc = vt.Memory(gpu, wc.nbytes).tensor([64, 32, 3], wc.tobytes())
mwl = vt.Memory(gpu, wl.nbytes).tensor([128, 64], wl.tobytes())

# one forward pass = one Graph
g = vt.Graph(rt, gpu)
g.enter()
x = g.input([64, 32], np.zeros((64, 32), np.float32).tobytes())   # [T, Cin]
h = vt.gelu(vt.conv1d(x, mwc, 1, 1, 1))                          # [T, 64]
y = vt.linear(h, mwl)                                            # [T, 128]
y.mark_output()                                                  # anything you read must be marked as output
out = np.frombuffer(y.to_bytes(), np.float32).reshape(64, 128)    # read = compute
g.exit()
```

Key points:

- Between `enter()` / `exit()` it's just **bookkeeping**; `to_bytes()` / `to_host()` is what actually computes.
- Any tensor you read must be `mark_output()`, or the scheduler will reuse its VRAM.
- `Memory` must be created **before** `enter()`; ops run on the **backend of the weights** (weights on GPU → ops on GPU).
- Ops are functions under `vt.`: `matmul` `linear` `conv1d` `rms_norm` `layer_norm` `gelu` `silu` `soft_max` `rope` `flash_attn` `snake_1d` `im2col_rafa` …
- Inputs via `g.input(shape, bytes)` (F32) / `g.input_i32(shape, bytes)`.
- **`shape()` eats the slowest dim's 1s** (at least one dim always remains):

  | PT shape | `shape()` |
  |---|---|
  | `(2, 1, 4)` | `(2, 1, 4)` |
  | `(1, 4)` | `(4,)` |
  | `(1, 2, 4)` | `(2, 4)` |

Reading weights from GGUF:

```python
f = vt.GgufFile(r"model.gguf", gpu)
w = f.tensor("some.weight")     # copies to VRAM, keeps the original quantized type
```

> float32 matmul goes through Vulkan's fp16 matrix cores, ~1e-3 off a double-precision reference — normal. It's the **device-level** default, shared by all models; to run full f32, set `GGML_VK_DISABLE_F16=1` **before** `import vulkantorch` (in C#, `SetEnvironmentVariable` before the first `vt_*` call). The `fp16:` line in the startup log shows the current state.

### 3. Reuse the Same Graph

If the shapes stay the same and only the data changes, a graph can be **allocated once**, then just re-fed new inputs — saving the repeated build + alloc inside a loop:

```python
g = vt.Graph(rt, gpu)
g.enter()
x = g.input([64, 32], np.zeros((64, 32), np.float32).tobytes())
y = vt.linear(vt.gelu(vt.conv1d(x, mwc, 1, 1, 1)), mwl)
y.mark_output()
g.exit()

g.alloc_static()                             # call once after building
for i in range(10):
    g.set_input(x, batch(i))                 # swap the input
    g.compute_static()                       # compute only, no re-alloc
    out = np.frombuffer(y.to_bytes(), np.float32).reshape(64, 128)
```

Key points:

- **`compute()` is one-shot**, only callable once; replay must use `alloc_static()` + `compute_static()`.
  (The scheduler re-allocates every time, which makes the result depend on *that* allocation pass, not the input — that's not a perf difference, it's **wrong numbers**.)
- A replayed graph must be **position-independent**: positions go in via `set_rows` as runtime inputs, attention windows pinned to a fixed size, masks as inputs too. See the bucketing in [GraphCache.cs](../VulkanTorch.Net/GraphCache.cs).
- For small models, who cares; but **autoregressive** cases that re-run per generated token — there it's real money (see [Speed Comparison](../README.en.md#three-language-speed-comparison)).

---

## C#: `VulkanTorch.Net`

### 1. Reference It

Not on NuGet. Project-reference it (or reference `VulkanTorch.Net.dll`), and drop the **native lib next to your exe**:

```xml
<ItemGroup>
  <ProjectReference Include="..\VulkanTorch.Net\VulkanTorch.Net.csproj" />
  <None Include="..\build\vulkantorch.dll" Link="vulkantorch.dll" CopyToOutputDirectory="PreserveNewest" />
</ItemGroup>
```

(ggml is statically linked in — this one file is all you need.)

### 2. Write Code

Same conv + linear:

```csharp
using VulkanTorch;

using var rt = new Runtime();
var gpu = rt.Gpu();

int T = 64, Cin = 32, Cout = 64, Lout = 128, K = 3;
var wc = new float[Cout * Cin * K];      // fill in your data (row-major)
var wl = new float[Lout * Cout];

using var mwc = new Memory(gpu, (ulong)wc.Length * 4);   // weights on the GPU
using var mwl = new Memory(gpu, (ulong)wl.Length * 4);
var twc = mwc.Tensor(new long[] { Cout, Cin, K }, Ops.F32, ToBytes(wc));
var twl = mwl.Tensor(new long[] { Lout, Cout }, Ops.F32, ToBytes(wl));

using var g = new Graph(rt, gpu);
g.Enter();
var x = g.Input(new long[] { T, Cin }, new float[T * Cin]);
var y = Ops.Linear(Ops.Gelu(Ops.Conv1d(x, twc, 1, 1, 1)), twl).MarkOutput();
var outv = y.ToFloats(g);                   // read = compute
g.Exit();

static byte[] ToBytes(float[] f)
{
    var b = new byte[f.Length * 4];
    Buffer.BlockCopy(f, 0, b, 0, b.Length);
    return b;
}
```

Same key points as Python: `Enter()`/`Exit()` is the capture scope, `ToFloats`/`ToBytes`/`ToInts` trigger compute; `MarkOutput()` anything you read; `Memory` created before `Enter()`; ops live in the `Ops` static class.

### 3. Reuse the Same Graph

```csharp
using var g = new Graph(rt, gpu);
g.Enter();
var x = g.Input(new long[] { T, Cin }, new float[T * Cin]);
var y = Ops.Linear(Ops.Gelu(Ops.Conv1d(x, twc, 1, 1, 1)), twl).MarkOutput();
g.Exit();

g.AllocStatic();                            // call once after building
for (int i = 0; i < 10; i++)
{
    g.SetInput(x, batch(i));                // swap the input
    g.ComputeStatic();                      // compute only, no re-alloc
    var outv = y.ToFloats(g);
}
```

Identical to the Python side: `Compute()` is one-shot, replay uses `AllocStatic()` + `ComputeStatic()`; the replayed graph must be position-independent. For autoregressive decode, just use [GraphCache.cs](../VulkanTorch.Net/GraphCache.cs)'s `BucketedReplay` (build once per window bucket, auto allocate + replay) — the model only provides two callbacks: "how to build the graph" and "what's the per-frame input".
