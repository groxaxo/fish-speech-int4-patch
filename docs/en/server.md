# Server

This page covers server-side inference for Fish Audio S2, plus quick links for WebUI inference and Docker deployment.

## API Server Inference

Fish Speech provides an HTTP API server entrypoint at `tools/api_server.py`.

### Start the server locally

```bash
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8080
```

Common options:

- `--compile`: enable `torch.compile` optimization
- `--half`: use fp16 mode
- `--api-key`: require bearer token authentication
- `--workers`: set worker process count
- `--max-seq-len`: reduce KV-cache preallocation for smaller GPUs

### Health check

```bash
curl -X GET http://127.0.0.1:8080/v1/health
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

For example:

```json
{
  "text": "Contact me at user@example.com and visit https://fish.audio/docs at 10:35 pm",
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
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "input": "Hello from Fish Speech.",
    "voice": "alloy",
    "response_format": "mp3",
    "stream": false
  }' --output speech.mp3
```

`voice` can either be:

- one of the standard OpenAI-compatible aliases such as `alloy` or `nova` (mapped to Fish Speech default synthesis), or
- an existing Fish Speech `reference_id`, which enables reference-conditioned synthesis through the OpenAI-style endpoint.

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
