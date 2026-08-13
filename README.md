<div align="center">
<h1>Fish Speech — 4 GB Ready BnB NF4 Fork</h1>

**English** | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [العربية](docs/README.ar.md) <br>

> **This is a community fork** of [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) that adds **bitsandbytes NF4 4-bit quantization** and a set of waste-removal optimisations, enabling inference on GPUs with as little as **4 GB** of VRAM.  
> Huge thanks to the amazing team at [Fish Audio](https://fish.audio/) for building and open-sourcing the original Fish Speech model — all credit for the core research and architecture belongs to them.  
> Built on top of [groxaxo/fish-speech-int4-patch](https://github.com/groxaxo/fish-speech-int4-patch), which introduced the bitsandbytes NF4 quantization path this fork extends down to 4 GB.

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
<br>
</div>
<br>

<div align="center">
    <a target="_blank" href="https://github.com/scarxity/fish-speech-int4-patch/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/scarxity/fish-speech-int4-patch?style=for-the-badge&label=Star%20the%20Fork"/>
    </a>
    <a target="_blank" href="https://huggingface.co/scarxity/fish-speech-s2-pro-nf4">
        <img alt="Hugging Face model" src="https://img.shields.io/badge/HuggingFace-scarxity%2Ffish--speech--s2--pro--nf4-f59e0b?style=for-the-badge"/>
    </a>
    <a target="_blank" href="https://github.com/fishaudio/fish-speech">
        <img alt="Upstream project" src="https://img.shields.io/badge/Upstream-fishaudio%2Ffish--speech-1f7a8c?style=for-the-badge"/>
    </a>
</div>

<div align="center">
    <strong>Run flagship S2-Pro on 4 GB cards, grab the NF4 model on Hugging Face, and if this saves you GPU pain, please star the fork.</strong>
</div>

<div align="center">
    <img src="https://count.getloli.com/get/@fish-speech?theme=asoul" /><br>
</div>

<br>

<div align="center">
    <a target="_blank" href="https://discord.gg/Es5qTB9BcN">
        <img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
    </a>
    <a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
    </a>
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/scarxity/fish-speech-s2-pro-nf4">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20NF4%20Model-scarxity%2Ffish--speech--s2--pro--nf4-orange"/>
    </a>
    <a target="_blank" href="https://github.com/scarxity/fish-speech-int4-patch/releases">
        <img alt="GitHub Releases" src="https://img.shields.io/badge/Releases-GitHub-1f7a8c?style=flat-square&logo=github&logoColor=white"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://github.com/fishaudio/fish-speech/blob/main/FishAudioS2TecReport.pdf">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **License Notice**  
> This codebase and its associated model weights are released under **[FISH AUDIO RESEARCH LICENSE](LICENSE)**. Please refer to [LICENSE](LICENSE) for more details. We will take action against any violation of the license.

> [!WARNING]
> **Legal Disclaimer**  
> We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Overview

This fork makes **Fish Speech S2-Pro** — a 4B-parameter flagship TTS model — run on
consumer GPUs, down to a **4 GB laptop card**, without giving up voice cloning,
multilingual support, or the inline expressiveness tags (`[laugh]`, `[whispers]`).

Measured on a simulated 4096 MiB budget with the `server-4gb` profile:

```
peak reserved : 3294 MiB of 4096 MiB   (3462 MiB on a 44 s utterance)
latency fit   : wall = 0.53s + 0.622 * audio_duration
```

The stock configuration cannot even finish loading at that budget. Everything the
4 GB profiles do is waste removal — none of it trades audio quality.

What you get:

- **bitsandbytes NF4 4-bit quantization** of the 4B backbone
- **Docker Compose profiles** for both card sizes, API and WebUI
- an **OpenAI-compatible API** on port `8880`, plus the native `/v1/tts` endpoint
- a **Gradio WebUI** on port `7860`
- **precomputed reference voices**, so cloning works without the codec encoder resident
- a bundled **default voice** for requests that specify no reference

Hardware guidance:

| Your GPU | Profile | Notes |
|---|---|---|
| 12–16 GB | `server` / `webui` | Full pipeline, fastest |
| 4–8 GB | `server-4gb` / `webui-4gb` | Everything below applies |

## Architecture

Three networks run per request. Knowing which does what explains every tuning knob
in this README.

**1. Slow AR backbone** — 36 layers, dim 2560 (~4B params, NF4). Emits one *semantic*
token per audio frame at **21.53 frames/second**. This is where words, pronunciation,
languages, and the inline emotion tags live. Untouched by any optimisation here.

**2. Fast (depth) AR** — 4 layers, dim 2560 (~400M params). Runs **nine times per
frame** to predict the residual codebooks that carry timbre and fine acoustic
texture. Because it repeats 9×, it is roughly half of all per-frame memory traffic.

**3. Codec** — a modified DAC. Takes the 10 codebook indices per frame and upsamples
×2048 to a 44.1 kHz waveform.

A request flows like this:

```
text ──▶ split at sentence boundaries (chunk_length)
     ──▶ build prompt: reference transcript + reference VQ codes + text
     ──▶ slow AR + fast AR generate 10 codes per frame, autoregressively
     ──▶ codec decodes codes to audio (in bounded chunks)
     ──▶ concatenate batches ──▶ WAV
```

Voice cloning is **in-context**, not fine-tuning: the reference clip's codes are
pasted into the prompt on every request, and the model continues in that voice.
That is why reference length is a per-request cost, and why ~10 s clips are
recommended over ~30 s ones.

### What the 4 GB profiles change

| Change | Saves | Why it is safe |
|---|---|---|
| `lm_head` projects onto 4097 rows, not 155776 | 0.72 GiB traffic/frame | Every other logit is masked to `-inf` before sampling — they were computed and discarded |
| Embedding table in host memory | 0.72 GiB | Only prefill embeds arbitrary vocabulary; generated tokens are always semantic |
| Codec loaded decode-only, fp16 | 1.36 GiB | Serving never touches the encoder |
| Codec mask sized from config | 1.0 GiB | Was hardcoded `32768²`, ignoring a declared `block_size` of 8192 |
| `max_seq_len` 2048 | 0.28 GiB | 32768 was inherited from a text-LLM config |
| Codec decoded in 64-frame chunks | decode stops scaling with length | Overlap-crop past the convolutions' 10-frame receptive field is exact |
| Weight norm folded at load | per-forward temporaries | Computed once instead of every call — bit-identical output |

## Setup with Docker

### Prerequisites

Docker with GPU support (Docker Desktop + WSL2 on Windows, or `nvidia-container-toolkit`
on Linux). Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

### 1. Clone and fetch the checkpoint

```bash
git clone https://github.com/scarxity/fish-speech-int4-patch
cd fish-speech-int4-patch

# ~4.9 GB
huggingface-cli download scarxity/fish-speech-s2-pro-nf4 --local-dir checkpoints/s2-pro
```

Use the **NF4** checkpoint. The unquantized release is 8.5 GB of bf16 that would have
to be quantized at load — more host RAM, slower startup, no benefit.

> [!IMPORTANT]
> Edit `checkpoints/s2-pro/tokenizer_config.json` and change
> `"tokenizer_class": "TokenizersBackend"` to `"PreTrainedTokenizerFast"`.
> `TokenizersBackend` is a transformers v5 name this loader does not recognise;
> without the edit the server dies with `UnboundLocalError: tokenizer`.
> `checkpoints/` is untracked, so re-downloading reverts this.

### 2. Precompute reference tokens

Do this **before the first start** — the 4 GB profiles load the codec without its
encoder, so they cannot turn reference audio into VQ codes at request time.

```bash
# every voice that is missing tokens
docker compose run --rm --entrypoint uv server-4gb \
  run --no-sync python tools/precompute_references.py
```

See [Voices and reference tokens](#voices-and-reference-tokens) for adding your own.

### 3. Start a profile

| Profile | Port | Command |
|---|---|---|
| `server-4gb` | 8880 | API, tuned for 4 GB |
| `webui-4gb` | 7860 | WebUI, tuned for 4 GB |
| `server` | 8880 | API, 12 GB+ |
| `webui` | 7860 | WebUI, 12 GB+ |

```bash
docker compose --profile server-4gb build
docker compose --profile server-4gb up -d
```

First start takes roughly 4–5 minutes (weight load, `torch.compile`, warm-up).
`/v1/health` returns 200 only once the model is ready, so it doubles as a readiness
probe.

> [!WARNING]
> Run **one profile at a time**. Each loads its own copy of the model, so two will
> not fit on a small card. Bring one down before starting another.

### 4. Send a request

```bash
curl -X POST http://localhost:8880/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world.", "reference_id": "my-voice"}' \
  --output out.wav
```

For anything longer than a couple of sentences, add `"max_new_tokens": 512`. The
prompt budget is `max_seq_len - max_new_tokens`, so lowering `max_new_tokens` widens
it at no memory cost — whereas raising `max_seq_len` costs both KV cache and compile
workspace. One request tops out around 800 characters; split longer text at
paragraph boundaries and send one request per paragraph.

> [!WARNING]
> The server has **no authentication**. Reach it over Tailscale or another private
> network rather than exposing the port.

### Environment variables

Set in `compose.yml` per profile, or overridden in a `.env` file.

| Variable | 4 GB default | Effect |
|---|---|---|
| `MAX_SEQ_LEN` | `2048` | KV cache and causal mask size |
| `FISH_OFFLOAD_EMBEDDINGS` | `1` | Keep the embedding table in host memory |
| `CODEC_DECODE_ONLY` | `1` | Drop the codec encoder, cast decode to fp16 |
| `FISH_DECODE_CHUNK_FRAMES` | `64` | Codec chunk size in frames; `0` disables |
| `COMPILE` | `1` | `torch.compile` the decode step |
| `ENABLE_LAZY_LOAD` | `0` | Load at startup rather than on first request |
| `API_PORT` / `GRADIO_PORT` | `8880` / `7860` | Published host ports |

## Voices and reference tokens

Each voice is a folder under `references/`:

```
references/<id>/sample.wav          the reference clip
references/<id>/sample.lab          its transcript, as plain text
references/<id>/sample.tokens.pt    precomputed VQ codes
```

`sample.wav` and `sample.lab` are yours to supply; `sample.tokens.pt` is generated:

```bash
docker compose run --rm --entrypoint uv server-4gb \
  run --no-sync python tools/precompute_references.py --reference-id <id>
```

Then select it per request with `"reference_id": "<id>"`. Omit the field and the
bundled default voice is used.

**Keep clips to about 10 seconds.** The reference is pasted into the prompt on every
request at 21.53 frames/second, so a 31.7 s clip costs 683 prompt tokens against 220
for a 10.2 s one — paid in KV cache and prefill every single time. It also matters
for precompute itself, whose activations scale with clip length:

| clip | peak VRAM |
|---|---|
| 10.2 s | 2254 MiB |
| 31.7 s | 3670 MiB — 26 MiB under a 4 GB cap |

Stop the running server before precomputing on a small card; it needs the GPU to
itself. For longer clips, precompute on a larger GPU and copy `sample.tokens.pt`
across. Copying a whole voice folder between machines works — the tokens are
portable.

The tool loads the **full** codec regardless of the profile's decode-only setting,
so `server` and `server-4gb` behave identically here.

### Published model

- Hugging Face model: [`scarxity/fish-speech-s2-pro-nf4`](https://huggingface.co/scarxity/fish-speech-s2-pro-nf4)
- Export helper: `python tools/llama/export_nf4.py --checkpoint-path checkpoints/s2-pro --output-path /tmp/s2-pro-nf4`

> [!NOTE]
> `--bnb4` targets the NF4 checkpoint. Do **not** point it at legacy `int4` or
> `int8` checkpoint directories.

## Quick Start

### Recommended docs

- [Running on a 4 GB GPU](docs/en/4gb-laptop.md) — the full small-card guide
- [12 GB install guide](docs/en/install.md)
- [Server guide](docs/en/server.md)
- [Command line inference](https://speech.fish.audio/inference/#command-line-inference)
- [WebUI inference](https://speech.fish.audio/inference/#webui-inference)
- [Docker setup](https://speech.fish.audio/install/#docker-setup)

> [!IMPORTANT]
> For SGLang server deployment, read the [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).

### For LLM agents

```text
Clone the repo, download scarxity/fish-speech-s2-pro-nf4 into checkpoints/s2-pro, and patch
tokenizer_config.json ("TokenizersBackend" -> "PreTrainedTokenizerFast"). Precompute
reference tokens, then start a Compose profile: server-4gb (4 GB cards) or server
(12 GB+), both on port 8880; webui-4gb / webui on 7860. Run one profile at a time.
Poll /v1/health for readiness. POST /v1/tts with {"text", "reference_id"}; add
"max_new_tokens": 512 for long text and split beyond ~800 characters at paragraph
boundaries. Do not use <|speaker:N|> tags - one voice per request via reference_id.
The canonical model name is `s2-pro`; OpenAI-style IDs `tts-1` and `tts-1-hd` work.
```

## Fish Audio S2  
**Best text-to-speech system among both open source and closed source**

Fish Audio S2 is the latest model developed by [Fish Audio](https://fish.audio/). Trained on over 10 million hours of audio across approximately 50 languages, S2 combines reinforcement learning alignment with a Dual-Autoregressive architecture to generate speech that sounds natural, realistic, and emotionally rich.

S2 supports fine-grained inline control of prosody and emotion using natural-language tags like `[laugh]`, `[whispers]`, and `[super happy]`.

> [!NOTE]
> Those tags are **plain text** — there is no parser and no escaping. They work
> because the training data paired them with the matching delivery, so a tag the
> model has not seen may simply be read aloud. Test a new tag before relying on it.
>
> This deployment serves **one voice per request** via `reference_id`, so the
> upstream `<|speaker:N|>` multi-speaker tokens have nothing to select and are not
> used. Send plain text; sentence splitting is handled server-side by
> `chunk_length`.

Visit the [Fish Audio website](https://fish.audio/) for live playground. Read the [blog post](https://fish.audio/blog/fish-audio-open-sources-s2/) and [technical report](https://github.com/fishaudio/fish-speech/blob/main/FishAudioS2TecReport.pdf) for more details.

### Model Variants

| Model | Size | Availability | Description |
|------|------|-------------|-------------|
| S2-Pro | 4B parameters | [HuggingFace](https://huggingface.co/scarxity/fish-speech-s2-pro-nf4) | NF4 build of the flagship model |

More details of the model can be found in the [technical report](https://arxiv.org/abs/2411.01156).

## Benchmark Results

| Benchmark | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER (Chinese) | **0.54%** (best overall) |
| Seed-TTS Eval — WER (English) | **0.99%** (best overall) |
| Audio Turing Test (with instruction) | **0.515** posterior mean |
| EmergentTTS-Eval — Win Rate | **81.88%** (highest overall) |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — Quality | **4.51 / 5.0** |
| Multilingual (MiniMax Testset) — Best WER | **11 of 24** languages |
| Multilingual (MiniMax Testset) — Best SIM | **17 of 24** languages |

On Seed-TTS Eval, S2 achieves the lowest WER among all evaluated models including closed-source systems: Qwen3-TTS (0.77/1.24), MiniMax Speech-02 (0.99/1.90), Seed-TTS (1.12/2.25). On the Audio Turing Test, 0.515 surpasses Seed-TTS (0.417) by 24% and MiniMax-Speech (0.387) by 33%. On EmergentTTS-Eval, S2 achieves particularly strong results in paralinguistics (91.61% win rate), questions (84.41%), and syntactic complexity (83.39%).

## Highlights

<img src="./docs/assets/totalability.png" width=200%>

### Fine-Grained Inline Control via Natural Language

S2 enables localized control over speech generation by embedding natural-language instructions directly at specific word or phrase positions within the text. Rather than relying on a fixed set of predefined tags, S2 accepts free-form textual descriptions — such as `[whisper in small voice]`, `[professional broadcast tone]`, or `[pitch up]` — allowing open-ended expression control at the word level.

### Dual-Autoregressive Architecture

S2 builds on a decoder-only transformer combined with an RVQ-based audio codec (10 codebooks, ~21 Hz frame rate). The Dual-AR architecture splits generation into two stages:

- **Slow AR** operates along the time axis and predicts the primary semantic codebook.
- **Fast AR** generates the remaining 9 residual codebooks at each time step, reconstructing fine-grained acoustic detail.

This asymmetric design — 4B parameters along the time axis, 400M parameters along the depth axis — keeps inference efficient while preserving audio fidelity.

### Reinforcement Learning Alignment

S2 uses Group Relative Policy Optimization (GRPO) for post-training alignment. The same models used to filter and annotate training data are directly reused as reward models during RL — eliminating distribution mismatch between pre-training data and post-training objectives. The reward signal combines semantic accuracy, instruction adherence, acoustic preference scoring, and timbre similarity.

### Production Streaming via SGLang

Because the Dual-AR architecture is structurally isomorphic to standard autoregressive LLMs, S2 directly inherits all LLM-native serving optimizations from SGLang — including continuous batching, paged KV cache, CUDA graph replay, and RadixAttention-based prefix caching.

On a single NVIDIA H200 GPU:

- **Real-Time Factor (RTF):** 0.195
- **Time-to-first-audio:** ~100 ms
- **Throughput:** 3,000+ acoustic tokens/s while maintaining RTF below 0.5

### Multilingual Support

S2 supports high-quality multilingual text-to-speech without requiring phonemes or language-specific preprocessing. Including:

**English, Chinese, Japanese, Korean, Arabics, German, French...**

**AND MORE!**

The list is constantly expanding, check [Fish Audio](https://fish.audio/) for the latest releases.

### Multi-Turn Generation

Thanks to the expansion of the model context, our model can now use previous information to improve the expressiveness of subsequent generated content, thereby increasing the naturalness of the content.

### Rapid Voice Cloning

Fish Audio S2 supports accurate voice cloning using a short reference sample (typically 10–30 seconds). The model captures timbre, speaking style, and emotional tendencies, producing realistic and consistent cloned voices without additional fine-tuning.
Please refer to [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md) to use the SGLang server.
---

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Tech Report
```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}
```
