# Fish Speech release bundle

This folder packages the rollout artifacts for the INT4 server work in one place.

## Included

- `reports/` — live benchmark and TTS-to-ASR correlation results
- `patches/` — patch files for the major commits in this rollout
- `scripts/` — helper scripts to start the server and rerun validation

## Major fixes included

- Kokoro-inspired text normalization and OpenAI-compatible `/v1` endpoints
- reduced hot-path overhead in semantic generation and VQ caching
- corrected `/v1/audio/speech` buffering/streaming behavior
- OpenAI-style model discovery and voice handling
- Spanish request autodetection that resolves the request language to `es` when Spanish text is detected, even if the client sends the wrong hint

## Notes

- Canonical benchmark and correlation helpers live in `tools/`.
- This bundle is for handoff and verification; the source of truth remains the normal project files in the repository root.
