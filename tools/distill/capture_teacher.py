#!/usr/bin/env python3
"""Capture teacher (hidden, codes) pairs for fast-transformer distillation.

Loads only the text2semantic LLM - no codec, no ModelManager, no server - runs
`generate_long` over `notes/distill/corpus.jsonl` in eager mode, and writes one
shard per utterance:

    hiddens: (T, 2560) float16   slow-model hidden state per generated frame
    codes:   (T, 10)   int16     sampled codebook indices, cb0 rebased to 0..4095
    meta:    dict                utt_id, text, lang, reference_id, temperature,
                                 seed, teacher

Eager mode is mandatory: `set_distill_capture_sink` is read inside
`decode_one_token_ar`, and a torch.compile'd decode specializes on the sink
being None at trace time.

    docker compose run --rm -T --no-deps \
      -e FISH_OFFLOAD_EMBEDDINGS=1 \
      -v "$PWD/notes:/app/notes" \
      --entrypoint uv server-4gb run --no-sync \
      python tools/distill/capture_teacher.py --limit 12

Resumable: an utterance whose shard already exists is skipped, so an
interrupted overnight run continues where it stopped. Per-utterance failures
are logged and skipped rather than killing the run.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import click
import pyrootutils
import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from loguru import logger  # noqa: E402

from fish_speech.models.text2semantic.inference import (  # noqa: E402
    generate_long,
    init_model,
    set_distill_capture_sink,
)

# Codec frame rate; only used to report hours of audio.
FRAME_RATE_HZ = 21.53

# Residual codebooks are physically 1024 entries even though the head is
# 4096-wide. Anything above this is worth reporting, not silently repairing.
RESIDUAL_LIMIT = 1023

TEACHER_NAME = "s2-pro-nf4"


def resolve_reference(
    reference_id: str, references_dir: Path
) -> tuple[torch.Tensor, str]:
    """Load precomputed VQ codes and transcript for a reference voice.

    Mirrors ReferenceLoader: every voice keeps its tokens under
    references/<id>/sample.tokens.pt, but the bundled "default" voice has its
    transcript at the repository root, not next to its cached tokens.
    """
    folder = references_dir / reference_id
    tokens_path = folder / "sample.tokens.pt"
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"No cached reference tokens at {tokens_path}. "
            "Run tools/precompute_references.py first."
        )

    text_path = folder / "sample.lab"
    if not text_path.exists():
        bundled = references_dir.parent / "sample.lab"
        if bundled.exists():
            text_path = bundled
        else:
            raise FileNotFoundError(f"No transcript for reference '{reference_id}'")

    tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
    transcript = text_path.read_text(encoding="utf-8").strip()
    logger.info(
        f"reference '{reference_id}': tokens {tuple(tokens.shape)} {tokens.dtype}, "
        f"transcript {text_path} ({len(transcript)} chars)"
    )
    return tokens, transcript


def capture_utterance(
    model,
    decode_one_token,
    record: dict,
    device: str,
    options: dict,
    reference: tuple[torch.Tensor, str],
    first: bool,
) -> dict:
    """Generate one utterance and return its shard payload."""
    prompt_tokens, prompt_text = reference
    num_codebooks = model.config.num_codebooks

    torch.manual_seed(int(record["seed"]))

    sink: list = []
    set_distill_capture_sink(sink)
    try:
        cursor = 0
        hidden_chunks: list[torch.Tensor] = []
        code_chunks: list[torch.Tensor] = []
        batches = 0

        for response in generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=record["text"],
            prompt_text=[prompt_text],
            prompt_tokens=[prompt_tokens],
            temperature=float(record["temperature"]),
            top_p=options["top_p"],
            repetition_penalty=options["repetition_penalty"],
            max_new_tokens=options["max_new_tokens"],
            chunk_length=options["chunk_length"],
            compile=False,
        ):
            if response.action != "sample":
                continue

            batches += 1
            emitted = response.codes  # (num_codebooks, n) on device
            n = int(emitted.size(1))
            captured = len(sink) - cursor

            # generate() runs one decode per emitted frame plus a terminal one
            # (the <|im_end|> frame, which it slices off). Every frame from
            # every batch belongs to this utterance, in order.
            if captured != n + 1:
                raise RuntimeError(
                    f"frame accounting mismatch in batch {batches}: sink grew by "
                    f"{captured}, generate emitted {n} codes (expected {n + 1})"
                )
            if first:
                logger.info(
                    f"frame accounting: sink +{captured} for {n} emitted frames "
                    "(1 terminal frame dropped)"
                )

            frames = sink[cursor : cursor + n]
            cursor += captured
            if not frames:
                continue

            hidden_chunks.append(torch.stack([h for h, _ in frames]))
            raw_codes = torch.stack([c for _, c in frames])  # (n, num_codebooks + 1)

            # Row 0 of the sink's code vector is the raw semantic token id; rows
            # 1.. are the audio codebooks with cb0 already rebased. That is the
            # same slicing generate() does (`y[1:, ...]`), so drop row 0 here and
            # the shard matches the production code stream exactly.
            if raw_codes.size(1) == num_codebooks + 1:
                audio_codes = raw_codes[:, 1:]
            elif raw_codes.size(1) == num_codebooks:
                audio_codes = raw_codes
            else:
                raise RuntimeError(
                    f"unexpected sink code width {raw_codes.size(1)} "
                    f"(num_codebooks={num_codebooks})"
                )

            expected = emitted.T.to(device="cpu", dtype=torch.int16)
            if not torch.equal(audio_codes, expected):
                bad = int((audio_codes != expected).sum())
                raise RuntimeError(
                    f"captured codes disagree with generate() output in batch "
                    f"{batches}: {bad} mismatched entries"
                )

            code_chunks.append(audio_codes)
    finally:
        set_distill_capture_sink(None)

    if not hidden_chunks:
        raise RuntimeError("no frames captured")

    hiddens = torch.cat(hidden_chunks).to(torch.float16).contiguous()
    codes = torch.cat(code_chunks).to(torch.int16).contiguous()

    frames = hiddens.size(0)
    if frames == 0 or codes.size(0) != frames:
        raise RuntimeError(
            f"length mismatch: hiddens {hiddens.shape} codes {codes.shape}"
        )
    if hiddens.size(1) != model.config.dim:
        raise RuntimeError(f"hidden width {hiddens.size(1)} != {model.config.dim}")
    if codes.size(1) != num_codebooks:
        raise RuntimeError(f"code width {codes.size(1)} != {num_codebooks}")

    semantic = codes[:, 0]
    if int(semantic.min()) < 0 or int(semantic.max()) >= model.config.codebook_size:
        raise RuntimeError(
            f"codebook 0 out of range: [{int(semantic.min())}, {int(semantic.max())}]"
        )
    if int(codes.min()) < 0:
        raise RuntimeError(f"negative code index: {int(codes.min())}")

    residual_over = int((codes[:, 1:] > RESIDUAL_LIMIT).any(dim=1).sum())

    return {
        "hiddens": hiddens,
        "codes": codes,
        "meta": {
            "utt_id": record["utt_id"],
            "text": record["text"],
            "lang": record["lang"],
            "reference_id": record["reference_id"],
            "temperature": float(record["temperature"]),
            "seed": int(record["seed"]),
            "teacher": TEACHER_NAME,
            "frames": frames,
            "batches": batches,
        },
        "_residual_over": residual_over,
    }


def save_shard(payload: dict, path: Path) -> int:
    """Write atomically so an interrupted run never leaves a half shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({k: v for k, v in payload.items() if not k.startswith("_")}, tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def append_index(index_path: Path, meta: dict, size_bytes: int) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {**meta, "bytes": size_bytes}
    with index_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_manifest(
    manifest_path: Path,
    index_path: Path,
    out_dir: Path,
    corpus_path: Path,
    failures: list[str],
) -> dict:
    """Aggregate every shard on disk, healing the index if it lags behind."""
    known: dict[str, dict] = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            known[entry["utt_id"]] = entry

    shards = sorted(out_dir.glob("*.pt")) if out_dir.exists() else []
    entries = []
    for shard in shards:
        utt_id = shard.stem
        entry = known.get(utt_id)
        if entry is None:
            blob = torch.load(shard, map_location="cpu", weights_only=False)
            entry = {**blob["meta"], "bytes": shard.stat().st_size}
            append_index(index_path, blob["meta"], shard.stat().st_size)
        entries.append(entry)

    total_frames = sum(int(e.get("frames", 0)) for e in entries)
    total_bytes = sum(int(e.get("bytes", 0)) for e in entries)
    manifest = {
        "corpus": str(corpus_path),
        "out_dir": str(out_dir),
        "teacher": TEACHER_NAME,
        "frame_rate_hz": FRAME_RATE_HZ,
        "shards": len(entries),
        "total_frames": total_frames,
        "total_hours": total_frames / FRAME_RATE_HZ / 3600,
        "total_bytes": total_bytes,
        "bytes_per_frame": (total_bytes / total_frames) if total_frames else 0.0,
        "by_lang": _tally(entries, "lang"),
        "by_reference_id": _tally(entries, "reference_id"),
        "by_temperature": _tally(entries, "temperature"),
        "failures": len(failures),
        "failed_utt_ids": failures[:200],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return manifest


def _tally(entries: list[dict], key: str) -> dict:
    counts: dict = {}
    for entry in entries:
        counts[str(entry.get(key))] = counts.get(str(entry.get(key)), 0) + 1
    return dict(sorted(counts.items()))


@click.command()
@click.option(
    "--corpus",
    type=click.Path(path_type=Path),
    default=Path("notes/distill/corpus.jsonl"),
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("notes/distill/shards"),
)
@click.option("--limit", type=int, default=0, help="stop after N utterances (0 = all)")
@click.option(
    "--start-index", type=int, default=0, help="skip the first N corpus lines"
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path),
    default=Path("checkpoints/s2-pro"),
)
@click.option(
    "--references-dir", type=click.Path(path_type=Path), default=Path("references")
)
@click.option("--device", type=str, default="cuda")
@click.option("--max-seq-len", type=int, default=2048)
@click.option("--max-new-tokens", type=int, default=512)
@click.option("--chunk-length", type=int, default=200)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
def main(
    corpus: Path,
    out_dir: Path,
    limit: int,
    start_index: int,
    checkpoint_path: Path,
    references_dir: Path,
    device: str,
    max_seq_len: int,
    max_new_tokens: int,
    chunk_length: int,
    top_p: float,
    repetition_penalty: float,
) -> None:
    records = [
        json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = records[start_index:]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise SystemExit(
            f"nothing to do: {len(records)} corpus lines, start-index {start_index}"
        )

    root = corpus.parent
    failures_path = root / "failures.log"
    index_path = root / "shard_index.jsonl"
    manifest_path = root / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"{len(selected)} utterances (of {len(records)}), out-dir {out_dir}, "
        f"offload_embeddings={os.getenv('FISH_OFFLOAD_EMBEDDINGS', '0')}"
    )

    load_start = time.perf_counter()
    model, decode_one_token = init_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        precision=torch.half,
        compile=False,
        max_length=max_seq_len,
        bnb4=True,
    )
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_len=max_seq_len, dtype=torch.half)
    model._cache_setup_done = True
    logger.info(f"model ready in {time.perf_counter() - load_start:.1f}s")

    options = {
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "chunk_length": chunk_length,
    }

    reference_cache: dict[str, tuple[torch.Tensor, str]] = {}
    failures: list[str] = []
    done = skipped = total_frames = residual_over = 0
    decode_seconds = 0.0
    first = True

    for position, record in enumerate(selected):
        utt_id = record["utt_id"]
        shard_path = out_dir / f"{utt_id}.pt"
        if shard_path.exists():
            skipped += 1
            continue

        try:
            reference_id = record["reference_id"]
            if reference_id not in reference_cache:
                reference_cache[reference_id] = resolve_reference(
                    reference_id, references_dir
                )

            started = time.perf_counter()
            payload = capture_utterance(
                model,
                decode_one_token,
                record,
                device,
                options,
                reference_cache[reference_id],
                first,
            )
            elapsed = time.perf_counter() - started
            first = False

            size_bytes = save_shard(payload, shard_path)
            append_index(index_path, payload["meta"], size_bytes)

            frames = payload["meta"]["frames"]
            done += 1
            total_frames += frames
            decode_seconds += elapsed
            residual_over += payload["_residual_over"]

            peak = (
                torch.cuda.max_memory_reserved() / 2**20
                if torch.cuda.is_available()
                else 0
            )
            logger.info(
                f"[{position + 1}/{len(selected)}] {utt_id} {record['lang']}/"
                f"{record['reference_id']} t={record['temperature']} "
                f"frames={frames} in {elapsed:.1f}s ({frames / elapsed:.2f} f/s) "
                f"{size_bytes / frames / 1024:.2f} KiB/frame | "
                f"done={done} frames={total_frames} "
                f"({total_frames / FRAME_RATE_HZ / 3600:.3f} h audio) "
                f"avg={total_frames / decode_seconds:.2f} f/s "
                f"vram_peak={peak:.0f} MiB"
            )
        except Exception as exc:  # noqa: BLE001 - an overnight run must not die here
            failures.append(utt_id)
            logger.error(f"{utt_id} failed: {exc}\n{traceback.format_exc()}")
            failure = {"utt_id": utt_id, "error": str(exc)}
            with failures_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest = write_manifest(manifest_path, index_path, out_dir, corpus, failures)

    logger.info(
        f"captured {done}, skipped {skipped}, failed {len(failures)}; "
        f"{total_frames} frames in {decode_seconds:.1f}s "
        f"({(total_frames / decode_seconds) if decode_seconds else 0:.2f} frames/s)"
    )
    if residual_over:
        logger.warning(
            f"{residual_over}/{total_frames} frames "
            f"({residual_over / max(total_frames, 1):.3%}) have a residual codebook "
            f"index above {RESIDUAL_LIMIT}"
        )
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / 2**20
        logger.info(f"VRAM peak reserved: {peak:.0f} MiB")
    logger.info(
        f"manifest: {manifest['shards']} shards, {manifest['total_frames']} frames, "
        f"{manifest['total_hours']:.3f} h, {manifest['total_bytes'] / 2**20:.1f} MiB, "
        f"{manifest['bytes_per_frame'] / 1024:.2f} KiB/frame"
    )


if __name__ == "__main__":
    main()
