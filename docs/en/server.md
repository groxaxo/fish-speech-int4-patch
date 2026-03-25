# Server

This page covers server-side inference for Fish Audio S2, plus quick links for WebUI inference and Docker deployment.

## Recommended default server

For this fork, the recommended path is the **12 GB BnB4 deployment**:

```bash
./install_bnb4_3060.sh
./start_bnb4_3060.sh
```

That default launcher gives you:

- `http://0.0.0.0:8880/v1`
- `--bnb4 --half`
- the same BnB4/FP16 defaults on the direct `tools/api_server.py` and `tools/run_webui.py` entrypoints
- lazy model loading on first inference
- automatic shutdown after **300 seconds** of inactivity
- port cleanup before startup if `8880` is already occupied

## API Server Inference

Fish Speech provides an HTTP API server entrypoint at `tools/api_server.py`.

### Start the server locally

```bash
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --lazy-load \
  --idle-timeout-seconds 300 \
  --listen 0.0.0.0:8880
```

`--bnb4` and `--half` are now the default for this fork, so you only need to pass them explicitly when you want to be extra clear. Use `--no-bnb4` or `--no-half` to opt out.

Common options:

- `--compile`: enable `torch.compile` optimization
- `--half`: use fp16 mode
- `--bnb4`: enable bitsandbytes NF4 4-bit quantization (requires `pip install bitsandbytes`; reduces VRAM to ~12 GB)
- `--lazy-load` / `--no-lazy-load`: control whether models load on first request or on server startup
- `--idle-timeout-seconds`: shut the server down after a period of inactivity (`0` disables the timeout)
- `--api-key`: require bearer token authentication
- `--workers`: set worker process count
- `--max-seq-len`: reduce KV-cache preallocation for smaller GPUs

### Health check

```bash
curl -X GET http://127.0.0.1:8880/v1/health
```

Expected response:

```json
{"status":"ok"}
```

### Main API endpoint

- `POST /v1/tts` for text-to-speech generation
- `POST /v1/audio/speech` for OpenAI-compatible text-to-speech requests
- `GET /v1/models` and `GET /v1/models/{id}` for OpenAI-compatible model discovery
- `GET /v1/audio/voices` for available Fish Speech reference IDs
- `POST /v1/vqgan/encode` for VQ encode
- `POST /v1/vqgan/decode` for VQ decode

### Text normalization

`/v1/tts` and `/v1/audio/speech` now support Kokoro-inspired text sanitization and normalization before inference. The request schema keeps the legacy `normalize` flag and also accepts a nested `normalization_options` object for finer control over:

- URL normalization
- email normalization
- optional pluralization (`(s)`)
- phone number normalization
- symbol replacement
- unit normalization

Both endpoints also accept an optional `language` hint. If the text clearly looks Spanish, the server automatically resolves the request language to `es` even when the caller sends the wrong hint. The resolved value is returned in the `X-Resolved-Language` response header.

For example:

```json
{
  "text": "Contact me at user@example.com and visit https://fish.audio/docs at 10:35 pm",
  "language": "en",
  "format": "wav",
  "normalize": true,
  "normalization_options": {
    "url_normalization": true,
    "email_normalization": true,
    "phone_normalization": true
  }
}
```

### OpenAI-compatible speech endpoint

Example request:

```bash
curl -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "s2-pro",
    "input": "Hello from Fish Speech.",
    "voice": "alloy",
    "language": "auto",
    "response_format": "mp3",
    "stream": false
  }' --output speech.mp3
```

`voice` can either be:

- one of the standard OpenAI-compatible aliases such as `alloy` or `nova` (mapped to Fish Speech default synthesis), or
- an existing Fish Speech `reference_id`, which enables reference-conditioned synthesis through the OpenAI-style endpoint.

Supported model IDs include the canonical `s2-pro` plus `tts-1`, `tts-1-hd`, and `fish-speech`.

If you send Spanish text such as `Hola señor, ¿cómo estás?` with `"language": "en"`, the server will still resolve the request to `es`.

## WebUI Inference

For WebUI usage, see:

- [WebUI Inference](https://speech.fish.audio/inference/#webui-inference)

## Docker

For Docker-based server or WebUI deployment, see:

- [Docker Setup](https://speech.fish.audio/install/#docker-setup)

You can also start the server profile directly with Docker Compose:

```bash
docker compose --profile server up
```

By default, the compose server profile now publishes the API on port `8880`.
