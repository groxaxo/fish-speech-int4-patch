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

## Run

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

## Known limitation: long utterances

**Audio beyond roughly 10–13 seconds runs out of memory during codec decode.**

Decoder activations scale with utterance length, because the decoder stack
expands to 44.1 kHz. Peak VRAM by output length:

| audio | peak |
|---|---|
| load only | 3030 MiB |
| 1.6 s | 3052 MiB |
| 5.9 s | 3394 MiB |
| 13.8 s | 3648 MiB — OOM |

Work around it by capping reply length, or by splitting text at sentence
boundaries client-side and issuing one request per sentence.

Two approaches that do **not** work, both verified:

- **Windowing the codec decode** with left context measures 12.2 dB SNR against
  a single-pass decode, with error spread evenly rather than concentrated at
  window boundaries. Causal does not imply a finite receptive field here:
  `post_module` is a `WindowLimitedTransformer` with `window_size: 128` and the
  decoder carries four transformer layers of its own. A working version would
  have to carry KV state across windows rather than merely prepend context.
- **Lowering `chunk_length`** has no effect. Splitting is driven by
  `<|speaker:N|>` tags, so plain text always logs
  `Split into 0 turns, grouped into 1 batches` and decodes in a single pass.

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

## If it still runs out of memory

The measurements above come from a simulated 4 GB card on a 16 GB one. Two
things that simulation cannot model: Windows' desktop compositor reserving part
of a real card, and the 3050's weaker compute.

In order of preference:

1. Drop `MAX_SEQ_LEN` to `1536` (safe with a 10 s reference).
2. Switch to the shorter reference clip if you are using a 30 s one.
3. Run the display off the integrated GPU so the 3050 is dedicated.
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
