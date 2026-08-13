# Running S2-Pro on a 4 GB GPU

Deployment guide for the `server-4gb` profile, tuned for an RTX 3050 Laptop
(4 GB). Measured on a simulated 4096 MiB budget:

```
peak reserved : 3294 MiB of 4096 MiB   (3462 MiB on a 44 s utterance)
latency fit   : wall = 0.53s + 0.622 * audio_duration
RTF           : short 0.89  medium 0.68  long 0.66   (steady-state)
```

The stock configuration cannot load at this budget at all — it runs out of
memory inside `.to(device)` at 3.54 GiB.

## Which checkpoint

Use the **NF4 quantized** checkpoint, `scarxity/fish-speech-s2-pro-nf4`.

Do *not* use the unquantized `s2-pro` release. It is 8.5 GB of bf16
safetensors that would have to be quantized at load: more host RAM, slower
startup, no benefit. It is only useful for future work that re-quantizes with
a different scheme (e.g. torchao int4), which is **not** needed to fit 4 GB.

## Setup

```bash
git clone <your-fork> fish-speech-int4-patch
cd fish-speech-int4-patch
git checkout perf/tier0-latency-vram

# ~4.9 GB
huggingface-cli download scarxity/fish-speech-s2-pro-nf4 --local-dir checkpoints/s2-pro
```

### Required checkpoint edit

Edit `checkpoints/s2-pro/tokenizer_config.json`:

```json
"tokenizer_class": "TokenizersBackend"   →   "PreTrainedTokenizerFast"
```

`TokenizersBackend` is a transformers v5 name that this fork's loader does not
recognise. Without the edit the server dies with `UnboundLocalError: tokenizer`
in `llama.py`.

`checkpoints/` is not tracked by git, so **re-downloading the checkpoint reverts
this every time**.

### Voices

Copy your `references/` directory across, including any `*.tokens.pt` files.
Each voice folder needs:

```
references/<id>/sample.wav    the reference clip
references/<id>/sample.lab    its transcript
references/<id>/sample.tokens.pt   precomputed VQ codes
```

Prefer a **~10 s reference over a ~30 s one**. Reference audio is pasted into
the prompt on every request at 21.53 frames/second, so a 31.7 s clip costs 683
prompt tokens against 220 for a 10.2 s clip. That is paid in both KV cache and
prefill on every single request.

To create a voice rather than copy one, see [Adding voices](#adding-voices).

## Run

### Precompute reference tokens

Run this before the first start, and again whenever you add a voice. The 4 GB
profiles load the codec without its encoder, so they cannot turn reference
audio into VQ codes at request time — the tokens have to exist on disk.

```bash
# every voice that is missing tokens
docker compose run --rm --entrypoint uv server-4gb \
  run --no-sync python tools/precompute_references.py

# or a single voice
docker compose run --rm --entrypoint uv server-4gb \
  run --no-sync python tools/precompute_references.py --reference-id <id>
```

This is a one-off command, not a server: it loads the *full* codec itself
regardless of the profile's decode-only setting, so `server` and `server-4gb`
behave identically here. Stop the running server first — at this budget the
precompute needs the GPU to itself. See [Adding voices](#adding-voices) for
the per-clip memory numbers.

### API server

```bash
docker compose --profile server-4gb build

# only if you did not copy the .tokens.pt files across
docker compose run --rm --entrypoint uv server-4gb \
  run --no-sync python tools/precompute_references.py

docker compose --profile server-4gb up -d
```

First start takes roughly 4–5 minutes (weight load, `torch.compile`, warm-up).
`/v1/health` returns 200 only once the model is ready, so it doubles as a
readiness probe.

### WebUI

```bash
docker compose --profile webui-4gb build
docker compose --profile webui-4gb up -d
```

Measured peak 3.18 GB. Roughly 5 minutes to start, of which ~116 s is
`torch.compile` — paid once at startup rather than on the first request.

**Run one or the other, never both.** Each profile loads its own copy of the
model, so two will not fit on a 4 GB card:

```bash
docker compose --profile server-4gb down
docker compose --profile webui-4gb up -d
```

Uploading reference audio through the WebUI does **not** work in this mode: the
codec has no encoder. Only voices already under `references/` with precomputed
tokens are selectable. See [Adding voices](#adding-voices).

## Network access

Both services bind `0.0.0.0` inside the container and Docker publishes on all
host interfaces.

| Service | Port | URL |
|---|---|---|
| API server | 8880 | `http://<host-ip>:8880/v1/tts` |
| WebUI | 7860 | `http://<host-ip>:7860` |

On Windows, add a firewall rule from an elevated PowerShell:

```powershell
New-NetFirewallRule -DisplayName "fish-speech" -Direction Inbound `
  -Protocol TCP -LocalPort 8880,7860 -Action Allow
```

**This server has no authentication.** Reach it over Tailscale or another
private network rather than exposing the port to an untrusted LAN. For a remote
client, use the host's Tailscale address (`tailscale ip -4`), not its LAN IP.

## Adding voices

Reference encoding needs the codec encoder, which the 4 GB profiles do not
load. `tools/precompute_references.py` loads the full codec itself and no
language model, so it still runs on a 4 GB card — but its activations scale
with clip length:

| clip | peak |
|---|---|
| 10.2 s | 2254 MiB |
| 31.7 s | 3670 MiB — 26 MiB under the cap |

So on a 4 GB card, **keep reference clips to about 10 seconds**. That is the
right choice anyway: a 31.7 s clip costs 683 prompt tokens on every request
against 220 for a 10.2 s one.

To add a voice:

```bash
# references/<id>/sample.wav + references/<id>/sample.lab
docker compose run --rm --entrypoint uv server-4gb \
  run --no-sync python tools/precompute_references.py --reference-id <id>
```

Stop the running server first — precompute needs the GPU to itself at this
budget. For longer clips, precompute on a larger GPU and copy the resulting
`sample.tokens.pt` across.

## Long text

Codec decode activations scale with utterance length, because the decoder stack
expands to 44.1 kHz — unchunked, a 13.8 s decode peaks at 3648 MiB and OOMs.
The 4 GB profiles therefore decode the codec's conv stack in 64-frame (~3 s)
chunks (`FISH_DECODE_CHUNK_FRAMES=64`): each chunk carries 32 frames of left
context that is cropped from the output, which is *exact* — causal convolutions
have a finite receptive field (measured: 10 frames), so beyond it the chunk
boundary has zero influence. Verified against single-pass decode down to the
float64 rounding floor (~295 dB SNR); in serving precision the chunked output
is as close to an fp32 reference as the unchunked one.

**For anything longer than a couple of sentences, send:**

```json
{
  "text": "...",
  "reference_id": "beatrice10",
  "max_new_tokens": 512
}
```

`chunk_length` can stay at its default of 200. Measured on a 578-character,
five-paragraph input under the 4096 MiB budget:

| `chunk_length` | wall | audio | RTF | peak |
|---|---|---|---|---|
| **200 (default)** | **24.3 s** | **38.1 s** | **0.64** | **3440 MiB** |
| 100 | 28.5 s | 39.7 s | 0.72 | 3462 MiB |
| 300 | 22.7 s | 43.7 s | 0.52 | 3462 MiB |
| 200, chunked decode off | 21.0 s | 38.1 s | 0.55 | 3700 MiB — 4 MiB under the cap |

The last row is why chunking is on by default: without it the same request
only survives by luck, and a real card driving a display does not have that
luck. The ~15% wall-time cost on long utterances is the overlap recompute.
Before chunked decode existed, this text OOM'd at `chunk_length` 200 and the
only working configuration was `chunk_length: 100` at RTF 1.13 — both the OOM
and the above-realtime RTF are gone.

### Why `max_new_tokens` and not `max_seq_len`

The prompt budget is `max_seq_len - max_new_tokens`. When that budget is too
small the request fails with `Prompt is too long: N > M`.

Raise the budget by **lowering `max_new_tokens`**, not by raising
`max_seq_len`. Both the KV cache and the `torch.compile` workspace scale with
`max_seq_len`, so raising it costs more memory than it buys:

| `max_seq_len` / `max_new_tokens` | load peak | result |
|---|---|---|
| 2048 / 1024 | 3030 MiB | prompt budget 1024 — too small at 4 batches |
| 4096 / 1024 | 3372 MiB | OOM during generation |
| 3072 / 512 | 3190 MiB | OOM during generation |
| **2048 / 512** | **3030 MiB** | **works** |

`max_new_tokens: 512` is ample per batch: a 100-byte batch is roughly 5 s of
speech, about 110 frames.

### The remaining ceiling

Each batch is generated with the previous batch's codes in context, so **the
prompt grows with total text length**, not with per-batch length. There is
therefore a limit on how much text one request can synthesize regardless of
`chunk_length`. Around 600 characters is comfortable; beyond that, split at
paragraph boundaries client-side and issue one request per paragraph, which
starts each with a fresh context.

### Why the chunking sits below `post_module`

An earlier attempt windowed the *whole* decode, `post_module` included, and
measured 12.2 dB SNR — audibly broken. `post_module` is an 8-layer
`WindowLimitedTransformer` (`window_size: 128`); stacking 8 window-128
attention layers gives a receptive field of ~1000 frames, far more left
context than was prepended. The working design runs `post_module` in one pass
— it operates at 21.5 Hz *before* upsampling, where a 30 s utterance is ~1 MB
of activations — and chunks only the conv stack below it, whose receptive
field is genuinely finite. (The config declares transformer layers inside the
decoder too, but `DecoderBlock` constructs and discards them without adding
them to its Sequential — the running decoder is pure causal convolution.)

## What the profile changes

Every item below is waste removal — none trades audio quality.

| Change | Saves | Why it is safe |
|---|---|---|
| `lm_head` projects onto 4097 rows, not 155776 | 0.72 GiB traffic/frame | `generate_long` masks all other logits to `-inf`; they were computed and discarded |
| Embedding table in host memory | 0.72 GiB | Only prefill embeds arbitrary vocabulary; generated tokens are always semantic or `im_end` |
| Codec loaded decode-only, fp16 | 1.36 GiB | Serving calls `from_indices()`, which never touches the encoder |
| Codec mask sized from `config.block_size` | 1.0 GiB | Was hardcoded `32768²`, ignoring a declared `block_size` of 8192 |
| `max_seq_len` 2048 | 0.28 GiB | Real worst case is ~1900 tokens; 32768 was inherited from a text-LLM config |
| Codec conv stack decoded in 64-frame chunks | ~250 MiB on long utterances, and decode stops scaling with length | Overlap-crop past the convs' 10-frame receptive field is exact; verified to the float64 rounding floor |
| Weight norm folded at load (all profiles) | per-forward weight temporaries | `w = g·v/‖v‖` computed once instead of every call — bit-identical output |

Reference encoding still runs at full fp32 in `tools/precompute_references.py`,
so cloned voices are bit-identical to the stock server.

## Environment variables

| Variable | Default in profile | Effect |
|---|---|---|
| `FISH_OFFLOAD_EMBEDDINGS` | `1` | Keep the embedding table in host memory |
| `CODEC_DECODE_ONLY` | `1` | Drop the codec encoder, cast decode path to fp16 |
| `MAX_SEQ_LEN` | `2048` | KV cache and causal mask size |
| `FISH_FULL_LM_HEAD` | unset | Set to `1` to restore the full projection (incompatible with the offload) |
| `FISH_DECODE_CHUNK_FRAMES` | `64` | Codec conv-stack chunk size in 21.5 Hz frames; `0` disables chunking |
| `FISH_DECODE_OVERLAP_FRAMES` | unset (code default 32) | Left context per chunk; exactness needs ≥ 10 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces allocator fragmentation |
| `API_PORT` / `GRADIO_PORT` | `8880` / `7860` | Published host ports |

## If it still runs out of memory

The measurements above come from a simulated 4 GB card on a 16 GB one. Two
things that simulation cannot model: Windows' desktop compositor reserving part
of a real card, and the 3050's weaker compute.

The desktop point is the big one: a card that is also driving a display starts
400–800 MB down before anything loads, against a measured peak of ~3.2–3.5 GB.

In order of impact:

1. **Run the display off the integrated GPU** so the discrete card is dedicated
   to compute. On a laptop this is usually a BIOS setting or a per-application
   preference in the NVIDIA control panel.
2. Switch to a ~10 s reference clip if you are using a longer one.
3. Drop `FISH_DECODE_CHUNK_FRAMES` to `32`, which halves the per-chunk decode
   activations for a little more overlap recompute.
4. Drop `MAX_SEQ_LEN` to `1536` (safe with a 10 s reference).
5. Reduce `--max-new-tokens` in the request, which lowers the KV high-water
   mark.

## Verifying a change yourself

`tools/vram_harness.py` simulates a small card on a large one via
`torch.cuda.set_per_process_memory_fraction()`, so 4 GB behaviour can be tested
on a bigger GPU:

```bash
FISH_OFFLOAD_EMBEDDINGS=1 uv run --no-sync python tools/vram_harness.py \
  --bnb4 --compile --codec-decode-only --budget-mib 4096 --max-seq-len 2048
```

It reports peak reserved VRAM and a least-squares fit of wall time against
audio duration, separating fixed overhead from marginal RTF. Note that the CUDA
context lives outside the allocator cap, which is why `--context-mib` (default
400) is subtracted from the budget before computing the fraction.

For codec changes there is also `tools/verify_codec_decode.py`, which compares
chunked against single-pass decode (and folded against unfolded weights) on
real reference tokens. Use `--sections float64` for exactness — fp32 has a
~4e-6 rounding floor of its own that has nothing to do with chunking.
