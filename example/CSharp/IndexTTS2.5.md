# IndexTTS 2.5 · C# 移植

**🌏 Language / 语言:** **中文** · [English](IndexTTS2.5.en.md)

[IndexTTS 2.5](https://github.com/index-tts/index-tts)（B站，零样本音色克隆 + 情绪控制）在
[vulkan-torch](../../README.md) 上的 C# 移植。整条推理链自包含，不依赖 Python 侧导出任何东西。

```
参考音频 ─┬─ fbank ─► CAMPPlus ─────────────────► style ──────────────┐
          ├─ fbank ─► Wav2Vec2-BERT ─► spk_cond ─┐                     │
          └─ 22.05k mel ────────────────────────┐│                     │
                                                 ▼▼                     ▼
文本 ─► 分词/归一化/分段 ─► GPT-2 AR ─► codec ─► 长度规整 ─► CFM(25 步 Euler+CFG) ─► BigVGAN ─► wav
```

（`prompt_condition` 和 s2mel 的 `cond` 都出自那个长度规整器：一个把参考音频的
`spk_cond` 拉长到 `ref_mel` 的长度，一个把 codec 的输出拉长到目标时长。）

---

## 1. 准备模型

需要一个 GGUF（约 3.3GB）和一份 tiktoken 词表。GGUF 由官方 4 个权重文件转出：

```bat
python tools\convert\convert_index_tts2_to_gguf.py --model <原始权重目录> ^
  --out model\indextts2.5\indextts2.5.f16.gguf
```

（原始权重：IndexTTS 2.5 的 `gpt.pth` / `codec.pth` / `s2mel.pth`，外加 `w2v/` `camplus/` `bigvan/`。
词表 `multilingual_zh_ja_yue_char_del.tiktoken` 已经入库，在 `model/indextts2.5/`。）

---

## 2. 命令行

```bat
dotnet build example\CSharp\IndexTts.Net -c Release

example\CSharp\IndexTts.Net\bin\Release\net10.0\IndexTts.Net.exe synth ^
  --model model\indextts2.5\indextts2.5.f16.gguf ^
  --ref-wav data\ref_audio\melinaref_24k.wav ^
  --text "大家好，这是一个测试。" ^
  --out data\indextts\out.wav
```

| 参数 | 说明 |
|---|---|
| `--model` | **必填**，IndexTTS 2.5 的 GGUF |
| `--tokenizer` | tiktoken 词表，默认 `model/indextts2.5/multilingual_zh_ja_yue_char_del.tiktoken` |
| `--ref-wav` | 参考音频（音色来源），默认 `data/ref_audio/melinaref_24k.wav` |
| `--text` | 要合成的文本 |
| `--out` | 输出 wav，默认 `data/indextts/cli.wav` |
| `--emo` | 情绪权重，如 `--emo "happy=0.6,calm=0.4"` |
| `--lang` | 语言代码，默认 `zh`（中/英/日/西等 99 种） |
| `--num-beams` | 束搜索宽度，默认 `1` |
| `--top-p` `--top-k` `--temperature` `--rep-penalty` | 采样参数，默认 `0.8 / 30 / 0.8 / 10` |
| `--cfg-rate` `--diffusion-steps` | CFM 的 CFG 强度与扩散步数，默认 `0.7 / 25` |
| `--duration-factor` | 语速/时长缩放，默认 `1.0` |
| `--seed` `--max-steps` | 采样种子 / 自回归步数上限（`0` = 按文本长度自动） |
| `--no-graph-cache` | 关掉 AR 的图缓存（A/B 对比用） |

**情绪**有 8 种，权重会自动做偏置并封顶总和 0.8：

```
happy  angry  sad  afraid  disgusted  melancholic  surprised  calm
```

**发音标注**直接写在 `--text` 里，比如 `今天天气<好|HAO3>不错。`

---

## 3. 在代码里用

每个阶段都是独立的类，输入输出统一是 **PyTorch 行主序布局**，激活走 `PT [T, C]`（时间在前）。
所有算子都要在 `Graph` 的 `Enter()`/`Exit()` 之间调用，读结果才算。

下面只是**骨架**（`/* ... */` 处省略了建图和参数细节），完整的可运行串联见 [Cli.cs](IndexTts.Net/Cli.cs)。

```csharp
using var rt = new Runtime();
var dev = rt.Gpu();
using var w = new GgufWeights(ggufPath, dev,
    new[] { "gpt.", "codec.", "s2mel.", "w2v.", "campplus.", "bigvgan.", "emo." });

// 文本 → 分段后的 token ids
var tok = new IndexTokenizer(tokenizerPath);
var front = new TextFrontend(tok);
var enc = front.EncodeForInference(text, 120, "zh");   // enc.Segments / enc.SegmentTokenIds

// 参考音频 → 音色 / 情绪条件（都依赖 host 侧的 DSP 前端）
var fb = Dsp.KaldiFbank(ToDouble(Resampler.Resample(wav, sr, 16000)), 16000, 80);
var feats = Dsp.Stack((float[,])fb.Clone(), out var mask);
var style = /* CampPlus.Forward(...) */
var spk   = /* W2vBert.Forward(feats, mask, mean, std) */

// AR 生成 mel codes
using var ar = new ArDecoder(rt, dev, w, 2048);
var logits = ar.Prefill(conds, ids, langId);           // conds = [3,1280] 软前缀
var codes  = ar.Generate(logits, ar.PromptLen, maxNew, rng);

// codes → 语义特征 → s2mel 条件 → CFM → 波形
var sinfer = new SemanticCodec(dev, w).Decode(g, codesT);          // [2T, 1024]
var cond   = new LengthRegulator(w).Forward(g, sinfer, targetLen); // [L, 512]
var mel    = new Cfm(rt, dev, dit).Inference(z, refMel, catCond, style, refMelLen, 0.7f, 25);
var wav    = new BigVgan(dev, w).Forward(g, melT);                 // [L*256, 1]
```

完整的串联见 [Cli.cs](IndexTts.Net/Cli.cs)——它就是个可读的端到端例子。

| 阶段 | 文件 | 入口 |
|---|---|---|
| tiktoken 分词 | [IndexTokenizer.cs](IndexTts.Net/IndexTokenizer.cs) | `Encode(text)` |
| 归一化 / 分段 / 发音标注 | [TextNormalization.cs](IndexTts.Net/TextNormalization.cs) · [TextFrontend.cs](IndexTts.Net/TextFrontend.cs) | `EncodeForInference(...)` |
| host DSP（FFT / fbank / mel） | [Dsp.cs](IndexTts.Net/Dsp.cs) | `KaldiFbank` `Stack` `MelSpectrogram` |
| 音色风格 | [CampPlus.cs](IndexTts.Net/CampPlus.cs) | `Forward(g, fbank)` |
| 语义编码器 | [W2vBert.cs](IndexTts.Net/W2vBert.cs) | `Forward(g, feats, mask, mean, std)` |
| 情绪条件 | [EmoCond.cs](IndexTts.Net/EmoCond.cs) · [EmotionVector.cs](IndexTts.Net/EmotionVector.cs) | `GetEmovec` / `Blend` |
| GPT-2 自回归 | [ArDecoder.cs](IndexTts.Net/ArDecoder.cs) · [BeamAr.cs](IndexTts.Net/BeamAr.cs) | `Prefill` / `Generate` |
| 语义 codec | [SemanticCodec.cs](IndexTts.Net/SemanticCodec.cs) | `Decode(g, codes)` |
| 时长规整 | [LengthRegulator.cs](IndexTts.Net/LengthRegulator.cs) | `Forward(g, x, ylens)` |
| 扩散（s2mel） | [Dit.cs](IndexTts.Net/Dit.cs) · [Cfm.cs](IndexTts.Net/Cfm.cs) | `Cfm.Inference(...)` |
| 声码器 | [BigVgan.cs](IndexTts.Net/BigVgan.cs) | `Forward(g, mel)` |

### 精度

权重是 f16，矩阵乘走 fp16 核，各阶段相对误差（对官方 PyTorch）：codec 2.3e-2、s2mel 5~8e-3、
BigVGAN 3.8e-2、Wav2Vec2-BERT 2.0e-2。冷启动约 5.5 秒（加载权重 2 秒 + 参考音频 0.4 秒 +
AR 0.5 秒 + 扩散/声码器 2 秒），合成 3.5 秒音频。

---

## 4. 待办

- **日语 / 西语里的阿拉伯数字**读不准。参考实现这两种语言走 NeMo 的 WFST 文本归一化（编译好的
  OpenFST 语法，不可移植），我们没做。中文和英文的数字是归一化过的，没问题。
- 情绪参考音频（参考实现的 `emo_audio_prompt`）只实现了"不给就用说话人音频"这条默认路径。
- 未做：文本→情绪（需要额外的 QwenEmotion 模型）、流式输出。
- `--num-beams 3` 在情绪条件下可能挑到超短序列（HF 按对数概率之和给束打分，短序列占优）。
