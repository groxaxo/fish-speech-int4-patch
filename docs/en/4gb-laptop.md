# Running S2-Pro on a 4 GB GPU

Deployment guide for the `server-4gb` profile, tuned for an RTX 3050 Laptop
(4 GB). Measured on a simulated 4096 MiB budget:

```
peak reserved : 3534 MiB of 4096 MiB
latency fit   : wall = 0.32s + 0.531 * audio_duration
RTF           : short 0.86  medium 0.60  long 0.58
```

The stock configuration cannot load at this budget at all — it runs out of
memory inside `.to(device)` at 3.54 GiB.

## Which checkpoint

Use the **NF4 quantized** checkpoint, `groxaxo/s2-pro-BnB-4Bits`.

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
huggingface-cli download groxaxo/s2-pro-BnB-4Bits --local-dir checkpoints/s2-pro
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
expands to 44.1 kHz. Generating a long passage in one pass exhausts a 4 GB card:

| audio in one decode | peak |
|---|---|
| load only | 3030 MiB |
| 1.6 s | 3052 MiB |
| 5.9 s | 3394 MiB |
| 13.8 s | 3648 MiB — OOM |

The fix is to keep each decode small by lowering `chunk_length`, which splits
the text at sentence boundaries. Each batch is generated with the previous
batch's codes in context, so prosody carries across the joins.

**For anything longer than a couple of sentences, send:**

```json
{
  "text": "...",
  "reference_id": "beatrice10",
  "chunk_length": 100,
  "max_new_tokens": 512
}
```

Measured on a 578-character, five-paragraph input (~37.6 s of speech):

| `chunk_length` | batches | result |
|---|---|---|
| 200 (default) | 4 | OOM |
| **100** | **9** | **OK — 42.5 s wall, RTF 1.13** |

Note the RTF above 1: many small batches cost more than one large one, since
per-batch overhead is paid repeatedly. That is the trade for fitting the
budget. Short replies still run at RTF 0.5–0.9 with default settings.

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

### An approach that does not work

**Windowing the codec decode** with left context measures 12.2 dB SNR against a
single-pass decode, with error spread evenly rather than concentrated at window
boundaries. Causal does not imply a finite receptive field here: `post_module`
is a `WindowLimitedTransformer` with `window_size: 128` and the decoder carries
four transformer layers of its own. A working version would have to carry KV
state across windows rather than merely prepend context.

## What the profile changes

Every item below is waste removal — none trades audio quality.

| Change | Saves | Why it is safe |
|---|---|---|
| `lm_head` projects onto 4097 rows, not 155776 | 0.72 GiB traffic/frame | `generate_long` masks all other logits to `-inf`; they were computed and discarded |
| Embedding table in host memory | 0.72 GiB | Only prefill embeds arbitrary vocabulary; generated tokens are always semantic or `im_end` |
| Codec loaded decode-only, fp16 | 1.36 GiB | Serving calls `from_indices()`, which never touches the encoder |
| Codec mask sized from `config.block_size` | 1.0 GiB | Was hardcoded `32768²`, ignoring a declared `block_size` of 8192 |
| `max_seq_len` 2048 | 0.28 GiB | Real worst case is ~1900 tokens; 32768 was inherited from a text-LLM config |

Reference encoding still runs at full fp32 in `tools/precompute_references.py`,
so cloned voices are bit-identical to the stock server.

## Environment variables

| Variable | Default in profile | Effect |
|---|---|---|
| `FISH_OFFLOAD_EMBEDDINGS` | `1` | Keep the embedding table in host memory |
| `CODEC_DECODE_ONLY` | `1` | Drop the codec encoder, cast decode path to fp16 |
| `MAX_SEQ_LEN` | `2048` | KV cache and causal mask size |
| `FISH_FULL_LM_HEAD` | unset | Set to `1` to restore the full projection (incompatible with the offload) |
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
3. Drop `MAX_SEQ_LEN` to `1536` (safe with a 10 s reference).
4. Reduce `--max-new-tokens` in the request, which lowers the KV high-water
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
