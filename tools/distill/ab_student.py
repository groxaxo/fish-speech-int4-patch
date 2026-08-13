"""Render the same sentences through the teacher and the distilled student.

Quality gate for the fast-transformer distillation. Both renders happen in one
process, from the same seed, with the model loaded once - same-seed output has
been observed to differ across container restarts, so a cross-process A/B would
compare two unrelated things.

Generation runs eager on purpose. ``decode_one_token_ar`` resolves
``model.fast_student`` once, so a torch.compile'd decode bakes the choice in at
trace time and swapping the attribute afterwards would silently do nothing.
Eager keeps the swap live; it costs wall time, which this tool does not measure.
Speed belongs to ``tools/vram_harness.py`` with and without FISH_FAST_STUDENT.

Sampling is autoregressive, so once the two runs pick different codes they
diverge for good - the pairs are for listening, not for diffing. ``--temperature
0.2`` makes sampling near-greedy and keeps them comparable much longer, which is
the setting to use when something sounds off and you want to localise it.

Usage::

    python tools/distill/ab_student.py --student notes/distill/ckpt/best.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import soundfile as sf
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fish_speech.models.dac.inference import load_model as load_codec  # noqa: E402
from fish_speech.models.text2semantic.fast_student import FastStudent  # noqa: E402
from fish_speech.models.text2semantic.inference import (  # noqa: E402
    generate_long,
    init_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_teacher import resolve_reference  # noqa: E402

# Companion-bot shaped: short reactions, an emotion tag, a quoted line, a
# multi-sentence paragraph, and both non-English languages the corpus covers.
DEFAULT_TEXTS = [
    ("short_en", "Welcome back. I missed you today."),
    ("tagged_en", "[laughs] You really did that? I can't believe you."),
    ("quoted_en", 'She looked at him and said, "I do not know you," and meant it.'),
    (
        "para_en",
        "It is strange, being remembered by no one. The world simply carries on, "
        "as though the space you occupied was never occupied at all. And yet he "
        "stays, holding on to someone who cannot hold him back.",
    ),
    ("short_id", "Halo sayang, hari ini aku bangun pagi sekali."),
    (
        "para_id",
        "Cuacanya cerah hari ini, jadi aku pikir kita bisa jalan-jalan sore nanti. "
        "Kalau kamu tidak terlalu sibuk dengan pekerjaanmu, tentu saja.",
    ),
    ("short_ja", "おかえりなさい。今日はどんな一日でしたか。"),
]


def render(
    model,
    decode_one_token,
    codec,
    text: str,
    reference,
    seed: int,
    temperature: float,
    device: str,
) -> tuple[torch.Tensor, "torch.Tensor", float]:
    """Generate codes for one text and decode them to audio."""
    prompt_tokens, prompt_text = reference

    torch.manual_seed(seed)
    start = time.perf_counter()

    chunks = []
    for response in generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        prompt_text=[prompt_text],
        prompt_tokens=[prompt_tokens],
        temperature=temperature,
        top_p=0.7,
        repetition_penalty=1.2,
        max_new_tokens=512,
        chunk_length=200,
        compile=False,
    ):
        if response.action == "sample":
            chunks.append(response.codes)

    if not chunks:
        raise RuntimeError(f"no codes generated for: {text[:40]!r}")

    codes = torch.cat(chunks, dim=1)
    gen_seconds = time.perf_counter() - start

    with torch.inference_mode():
        audio = codec.from_indices(codes[None].long())[0].squeeze()

    return codes, audio.float().cpu(), gen_seconds


@click.command()
@click.option("--student", type=Path, required=True, help="FastStudent checkpoint.")
@click.option("--llama-path", type=Path, default=Path("checkpoints/s2-pro"))
@click.option("--codec-path", type=Path, default=Path("checkpoints/s2-pro/codec.pth"))
@click.option("--codec-config", default="modded_dac_vq")
@click.option("--references-dir", type=Path, default=Path("references"))
@click.option("--reference-id", default="beatrice10")
@click.option("--out", type=Path, default=Path("notes/distill/ab"))
@click.option("--seed", type=int, default=12345, show_default=True)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    show_default=True,
    help="Production default. Use 0.2 to keep the two runs comparable.",
)
@click.option("--max-seq-len", type=int, default=2048, show_default=True)
@click.option("--device", default="cuda")
def main(
    student: Path,
    llama_path: Path,
    codec_path: Path,
    codec_config: str,
    references_dir: Path,
    reference_id: str,
    out: Path,
    seed: int,
    temperature: float,
    max_seq_len: int,
    device: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    precision = torch.half

    logger.info("loading teacher (eager)")
    model, decode_one_token = init_model(
        checkpoint_path=llama_path,
        device=device,
        precision=precision,
        compile=False,
        max_length=max_seq_len,
        bnb4=True,
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1, max_seq_len=max_seq_len, dtype=precision
        )
    model._cache_setup_done = True

    # init_model honours FISH_FAST_STUDENT; this tool drives the swap itself so
    # both sides come from one process, so start from a clean teacher.
    model.fast_student = None

    logger.info(f"loading student from {student}")
    fast_student = FastStudent.load(student, device=device, dtype=precision)
    fast_student.setup_caches(1, dtype=precision, device=torch.device(device))
    logger.info(
        f"student: {fast_student.num_parameters / 1e6:.1f}M params, "
        f"dim {fast_student.config.dim}"
    )

    logger.info("loading codec (decode-only)")
    codec = load_codec(
        config_name=codec_config,
        checkpoint_path=codec_path,
        device=device,
        decode_only=True,
        precision=precision,
    )

    reference = resolve_reference(reference_id, references_dir)
    sample_rate = codec.sample_rate

    logger.info(
        f"rendering {len(DEFAULT_TEXTS)} texts x 2 models, seed {seed}, "
        f"temperature {temperature}"
    )
    rows = []
    for name, text in DEFAULT_TEXTS:
        outputs = {}
        for label in ("teacher", "student"):
            model.fast_student = fast_student if label == "student" else None
            codes, audio, gen_seconds = render(
                model,
                decode_one_token,
                codec,
                text,
                reference,
                seed,
                temperature,
                device,
            )
            path = out / f"{name}_{label}.wav"
            sf.write(path, audio.numpy(), sample_rate)
            outputs[label] = (codes, audio, gen_seconds)
            logger.info(
                f"  {name:10s} {label:7s} {codes.shape[1]:4d} frames -> {path.name}"
            )

        t_codes = outputs["teacher"][0]
        s_codes = outputs["student"][0]
        shared = min(t_codes.shape[1], s_codes.shape[1])
        # How far the two stay identical before sampling noise separates them.
        # Informative at low temperature, near-meaningless at 0.7.
        same = (t_codes[:, :shared] == s_codes[:, :shared]).all(dim=0)
        prefix = int(same.cumprod(0).sum().item()) if shared else 0

        rows.append(
            (
                name,
                t_codes.shape[1],
                s_codes.shape[1],
                outputs["teacher"][1].shape[-1] / sample_rate,
                outputs["student"][1].shape[-1] / sample_rate,
                prefix,
            )
        )

    header = (
        f"{'case':12s} {'T frames':>9s} {'S frames':>9s} "
        f"{'T secs':>7s} {'S secs':>7s} {'same prefix':>12s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, tf, sf_, td, sd, prefix in rows:
        print(f"{name:12s} {tf:9d} {sf_:9d} {td:7.2f} {sd:7.2f} {prefix:12d}")

    print(f"\nwavs in {out}/  - listen to <case>_teacher.wav vs <case>_student.wav")
    print(
        "Frame counts and durations differing is expected: sampling is "
        "autoregressive, so the runs diverge once they pick different codes."
    )


if __name__ == "__main__":
    main()
