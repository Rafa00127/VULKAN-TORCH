# HiggsTTS v3 · 移植

**🌏 Language / 语言:** **中文** · [English](higgstts.en.md)

[HiggsTTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)（4B TTS）在
[vulkan-torch](../README.md) 上的移植，两个库各实现了一份：

| | 模型代码 | 可执行入口 |
|---|---|---|
| **Python** | [`python/higgstts_vt/`](../example/python/higgstts_vt/)：`model.py` / `ar.py` / `decode.py` / `tts.py` | `cli.py` |
| **C#** | [`CSharp/HiggsTtsSharp/`](../example/CSharp/HiggsTtsSharp/)：`EncodeRef.cs` / `Ar.cs` / `DacDecoder.cs` / `HiggsTokenizer.cs` / `Resampler.cs` | [`example/CSharp/HiggsTts.Net/`](../example/CSharp/HiggsTts.Net/)（`Cli.cs`） |

两份都自包含（C# 自带分词器和重采样器，不需要 Python 侧导出任何东西），互不依赖。

---

## 用法

两边都是 **`--model` 必填**，其余参数同名、含义一致（完整参数表在根 [README](../README.md)，
加 `-h` 也能看）。

**Python** —— [`cli.py`](../example/python/higgstts_vt/cli.py)：

```bat
python example\python\higgstts_vt\cli.py ^
  --model path\to\HiggsTTS3-q8_0.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

**C#** —— [`HiggsTts.Net`](../example/CSharp/HiggsTts.Net/)：

```bat
dotnet build example\CSharp\HiggsTts.Net -c Release

example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe synth ^
  --model D:\models\HiggsTTS3-q8_0.gguf ^
  --ref-text "I have no doubt you will become Elden Lord, may you take the throne." ^
  --text "<|style:whispering|>Hello how you doing? Are you having fun these days?"
```

`--ref-wav`（默认 `data/ref_audio/melinaref_24k.wav`）、`--ref-text`、`--text`、`--out`
都有默认值，所以最小调用就是 `--model` 加一句 `--text`。几个不同的地方：

| | Python | C# |
|---|---|---|
| 只跑 encode_ref | `--encode-only` | 模式参数 `encode`（不合成，输出 `<out>.i32` 的码） |
| `--out` 默认 | `data/higgstts/cli_py.wav` | `data/higgstts/cli_cs_synth.wav` |
| tokenizer | 默认用仓库里的 `data/ref_audio/higgs_tts_v3_tokenizer.json` | 不给 `--tokenizer` 就从 GGUF 的 vocab/merges 现建 |

采样参数 `--temperature`（0.9）/ `--topk`（50）/ `--seed`（42）/ `--max-steps`（0 = 按文本长度自动）
和 `--no-graph-cache` 两边同名同义。

**流式 HTTP 接口**（C# 的 `serve` 模式），OpenAI 兼容、边生成边出声：

```bat
example\CSharp\HiggsTts.Net\bin\Release\net10.0\HiggsTts.Net.exe serve ^
  --model D:\models\HiggsTTS3-q8_0.gguf --port 8000 ^
  --ref-wav data\ref_audio\melinaref_24k.wav --ref-text "I have no doubt you will become Elden Lord." ^
  --voice alice=D:\voices\alice.wav

REM 另一个 shell：
curl -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" ^
  -d "{\"input\":\"Hello there.\",\"voice\":\"default\",\"response_format\":\"pcm\"}" -o out.pcm
```

`POST /v1/audio/speech`

`--voice 名字=<wav>[,<文字稿>]` 注册额外音色

---

## 链路

```
参考音频 ──► encode_ref ──► RVQ 码 ──┐(码长：给参考音频留占位符)
             (codec)                │
                                     └──────────┐
                                                ▼
文本/参考文本 ──► tokenizer ──► ids ──► build_prompt ──► prompt ──┐
                                                                  ├──► Qwen3 36层 AR ──► 码 ──► DAC 解码 ──► wav
                       RVQ 码（码本身，和 prompt 一起喂进 AR）─────┘        (backbone)         (decoder)
```

---

## 权重

[NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf)（约 4GB），或自己转：

```bat
python tools\convert\convert-higgs-tts-to-gguf.py --input <HF safetensors 目录> --output higgs.gguf
```

**F16 转换有个坑**（脚本已经处理，自己改的时候别踩）：1-D 张量（RMSNorm 权重、bias、snake-alpha）
和 10 个 `conv_t` wperm 2-D 张量必须留在 **F32**。降成 F16 会让加载器按 F32 的字节数去读，
直接 `GGML_ASSERT(tensor read out of bounds)` —— 即 `--outtype f16` 只该量化 2-D 权重，
不是"全都降 F16"。

---

## 两个实现的对齐情况

DAC 解码和 AR 的 logits **逐位一致**，tokenizer 输出也一致；`encode_ref` 用同一个重采样器时
97% 的帧相同（差异只来自 Python 用 librosa、C# 用自带 Kaiser 重采样，听感无差别）。

速度约 **0.2~0.3 RTF** （在我的7900 xtx上）：Decode 几乎免费（41 ms，耗时全在 36 层 backbone AR）。
另外两个模型的简介见 [example-models.md](example-models.md)。
