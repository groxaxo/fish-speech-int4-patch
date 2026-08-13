"""Prove that chunked codec decoding reproduces a single pass.

`DAC.from_indices(..., chunk_frames=N)` runs the convolutional half of decoding
over slices instead of the whole utterance, which is what keeps activation
memory bounded on a small card. Everything below `post_module` is causal
convolution with a finite receptive field, so a chunk fed enough real left
context - and then stripped of that context's audio - must reproduce the
single-pass result exactly.

"Exactly" is a statement about the arithmetic, not about the bits a particular
BLAS produces. Convolution kernels pick different blocking and vectorisation
for different input lengths, so a chunk and a full pass round differently even
when they compute the same sum. That shows up as a floor of ~4e-6 in float32 on
CPU - 114 dB, four orders of magnitude below the 8e-2 error you get from
chunking with no context at all, but not zero.

So on CPU this script runs each comparison twice:

  float32  the arithmetic at serving precision, judged by SNR.
  float64  the exactness proof. Rounding drops to ~1e-15, so a genuine
           truncation error stands out by orders of magnitude. This is also
           where the minimum viable overlap is measured, because in float32 the
           outer taps of the receptive field are quieter than the noise floor
           and cannot be seen at all.

    python tools/verify_codec_decode.py --device cpu
    python tools/verify_codec_decode.py --device cuda

--device cuda runs the fp16 decode-only path the 4 GB server actually uses.
fp16 rounding is coarse enough (~54 dB) that comparing chunked fp16 against
unchunked fp16 measures nothing but fp16, so that mode scores both against an
fp32 reference and asks whether chunking costs anything on top.

Exits nonzero on failure.
"""

import math
from pathlib import Path

import click
import pyrootutils
import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.dac.inference import load_model

# Audio samples produced per token frame: 4 from quantizer.upsample, 512 from
# the decoder. A chunk of C frames holds C * 2048 samples of activations per
# layer instead of the whole utterance.
SAMPLES_PER_FRAME = 2048
FRAME_RATE = 44100 / SAMPLES_PER_FRAME  # 21.53 Hz

# Must hold. Overlap 32 is the shipped default; the odd chunk sizes exercise a
# short final chunk and lengths that do not divide evenly.
REQUIRED_COMBOS = [(64, 32), (64, 64), (37, 32), (50, 32), (128, 32)]

# Reported, not required: these probe at or below the receptive field.
DIAGNOSTIC_COMBOS = [(64, 0), (64, 4), (64, 8), (64, 16)]

# float64 is slow, so its combo list is trimmed to the two that matter: the
# default, and one whose chunk size divides nothing.
EXACT_COMBOS = [(64, 32), (37, 32)]

# Pass thresholds per dtype. float64 asserts the arithmetic; the lower
# precisions can only assert that nothing beyond rounding went wrong.
MAX_DIFF_FLOAT64 = 1e-12
MIN_SNR_FLOAT32 = 100.0

DEFAULT_OVERLAP = 32  # DAC.from_indices default


def snr_db(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Signal-to-noise ratio of candidate against reference, in dB."""
    reference = reference.double()
    candidate = candidate.double()
    noise = (reference - candidate).pow(2).sum().item()
    if noise == 0.0:
        return math.inf
    signal = reference.pow(2).sum().item()
    if signal == 0.0:
        return -math.inf
    return 10.0 * math.log10(signal / noise)


def load_cases(references_dir: Path, device: str):
    """Real reference tokens where available, random indices otherwise."""
    cases = []
    for name in ("beatrice10", "beatrice30"):
        path = references_dir / name / "sample.tokens.pt"
        if not path.exists():
            click.echo(f"  missing {path}, skipping")
            continue
        codes = torch.load(path, map_location="cpu", weights_only=True)
        if codes.ndim == 2:
            codes = codes[None]
        cases.append((name, codes.long().to(device)))

    if not cases:
        click.echo("  no reference tokens found, falling back to random indices")
        generator = torch.Generator().manual_seed(0)
        for name, frames in (("random220", 220), ("random683", 683)):
            codes = torch.randint(0, 1024, (1, 10, frames), generator=generator)
            cases.append((name, codes.long().to(device)))

    return cases


@torch.inference_mode()
def decode(model, indices: torch.Tensor, chunk_frames: int, overlap_frames: int):
    # decode_codes clamps indices in place; hand it a copy so repeated runs on
    # the same case stay independent.
    return model.from_indices(
        indices.clone(),
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
    )


def compare(model, indices, chunk, overlap, single=None):
    reference = decode(model, indices, 0, 0) if single is None else single
    chunked = decode(model, indices, chunk, overlap)
    return (reference - chunked).abs().max().item(), snr_db(reference, chunked)


def run_combos(model, cases, combos, judge, label):
    """Compare single-pass against each (chunk, overlap); return failures."""
    failures = []
    for name, indices in cases:
        frames = indices.shape[-1]
        single = decode(model, indices, 0, 0)
        expected = frames * SAMPLES_PER_FRAME
        click.echo(
            f"\n  {name}: {frames} frames -> {single.shape[-1]} samples "
            f"({single.shape[-1] / 44100:.2f}s)"
        )
        if single.shape[-1] != expected:
            failures.append(
                f"[{label}] {name}: single pass produced {single.shape[-1]} "
                f"samples, expected {expected}"
            )

        for chunk, overlap in combos:
            if chunk >= frames:
                click.echo(
                    f"    chunk {chunk:4d} overlap {overlap:3d}: skipped (T <= chunk)"
                )
                continue
            diff, snr = compare(model, indices, chunk, overlap, single=single)
            # judge returns None for combos that are reported but not required.
            ok = judge(chunk, overlap, diff, snr)
            click.echo(
                f"    chunk {chunk:4d} overlap {overlap:3d}: "
                f"max|diff| {diff:.3e}  SNR {snr:8.2f} dB"
                + ("  (diagnostic)" if ok is None else ("  OK" if ok else "  FAIL"))
            )
            if ok is False:
                failures.append(
                    f"[{label}] {name}: chunk {chunk} overlap {overlap}: "
                    f"max|diff| {diff:.3e}, SNR {snr:.2f} dB"
                )
    return failures


def run_fp16_combos(model, reference_model, cases, min_snr_db, max_drop_db):
    """Judge the fp16 serving path against an fp32 CPU reference.

    fp16 carries ~11 bits of mantissa, so the same arithmetic in two different
    orderings lands on visibly different values - around 54 dB apart, no matter
    how much context a chunk is given. Comparing chunked fp16 against unchunked
    fp16 therefore only measures fp16. The question worth asking is whether
    chunking moves the result any further from the true answer than fp16 has
    already moved it, so both are scored against the same fp32 reference.
    """
    failures = []
    for name, indices in cases:
        frames = indices.shape[-1]
        reference = decode(reference_model, indices.cpu(), 0, 0).float()
        single = decode(model, indices, 0, 0)
        base_snr = snr_db(reference, single.float().cpu())
        click.echo(
            f"\n  {name}: {frames} frames -> {single.shape[-1]} samples "
            f"({single.shape[-1] / 44100:.2f}s)"
        )
        click.echo(
            f"    unchunked fp16 vs fp32 reference: SNR {base_snr:7.2f} dB "
            "  <- the cost of fp16 alone"
        )
        if base_snr < min_snr_db:
            failures.append(
                f"[float16] {name}: unchunked fp16 is only {base_snr:.2f} dB "
                f"from the fp32 reference"
            )

        for chunk, overlap in REQUIRED_COMBOS + DIAGNOSTIC_COMBOS:
            if chunk >= frames:
                click.echo(
                    f"    chunk {chunk:4d} overlap {overlap:3d}: skipped (T <= chunk)"
                )
                continue
            chunked = decode(model, indices, chunk, overlap)
            snr_ref = snr_db(reference, chunked.float().cpu())
            snr_single = snr_db(single, chunked)
            required = (chunk, overlap) in REQUIRED_COMBOS
            ok = snr_ref >= base_snr - max_drop_db if required else None
            click.echo(
                f"    chunk {chunk:4d} overlap {overlap:3d}: "
                f"vs fp32 {snr_ref:7.2f} dB "
                f"({snr_ref - base_snr:+6.2f} dB vs unchunked)  "
                f"vs fp16 {snr_single:7.2f} dB"
                + ("  (diagnostic)" if ok is None else ("  OK" if ok else "  FAIL"))
            )
            if ok is False:
                failures.append(
                    f"[float16] {name}: chunk {chunk} overlap {overlap}: "
                    f"{base_snr - snr_ref:.2f} dB worse than unchunked fp16"
                )
    return failures


def find_min_exact_overlap(model, indices, chunk, max_overlap, floor_overlap):
    """Smallest overlap whose chunked decode matches a single pass.

    The bar is set by measuring the residual at an overlap far beyond any
    plausible receptive field: whatever difference survives there is pure
    rounding, and anything at that level is indistinguishable from exact.
    Scans upward rather than bisecting - the whole curve is the interesting
    part - and confirms two further overlaps before believing a result.
    """
    single = decode(model, indices, 0, 0)
    floor, _ = compare(model, indices, chunk, floor_overlap, single=single)
    tolerance = max(4.0 * floor, 1e-15)
    click.echo(
        f"    rounding floor at overlap {floor_overlap}: {floor:.3e} "
        f"-> calling anything <= {tolerance:.3e} exact"
    )

    found = None
    for overlap in range(0, max_overlap + 1):
        diff, _ = compare(model, indices, chunk, overlap, single=single)
        exact = diff <= tolerance
        click.echo(
            f"    overlap {overlap:3d}: max|diff| {diff:.3e}"
            + ("  <- exact" if exact else "")
        )
        if exact:
            if found is None:
                found = overlap
            if overlap >= found + 2:
                return found
        else:
            found = None
    return found


@click.command()
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/s2-pro-nf4/codec.pth",
    type=click.Path(exists=True),
)
@click.option("--device", default="cpu", help="cpu (fp32 + fp64) or cuda (fp16)")
@click.option("--references-dir", default="references", type=click.Path(path_type=Path))
@click.option(
    "--min-snr-db",
    default=45.0,
    help="sanity floor: how close unchunked fp16 must sit to the fp32 reference",
)
@click.option(
    "--max-snr-drop-db",
    default=1.0,
    help="how much SNR chunking may cost the fp16 path relative to unchunked",
)
@click.option("--search-frames", default=96, help="frames used for the overlap search")
@click.option("--max-overlap-search", default=20)
@click.option(
    "--floor-overlap",
    default=64,
    help="overlap assumed to be far beyond the receptive field",
)
@click.option(
    "--sections",
    default="float32,float64,overlap",
    help="which CPU sections to run; float64 is slow",
)
def main(
    config_name,
    checkpoint_path,
    device,
    references_dir,
    min_snr_db,
    max_snr_drop_db,
    search_frames,
    max_overlap_search,
    floor_overlap,
    sections,
):
    cuda = device.startswith("cuda")
    wanted = {s.strip() for s in sections.split(",") if s.strip()}
    # decode_only drops the encoder, which from_indices never touches; on CPU
    # that is just less RAM and a faster load. Casting is only legal alongside
    # decode_only, and fp16 is what the 4 GB server runs.
    model = load_model(
        config_name,
        checkpoint_path,
        device=device,
        decode_only=True,
        precision=torch.half if cuda else None,
    )

    cases = load_cases(references_dir, device)
    if not cases:
        raise SystemExit("no test cases")

    failures = []

    if cuda:
        click.echo(
            f"\n== float16 on {device}: no worse than unchunked fp16 "
            f"by more than {max_snr_drop_db} dB =="
        )
        reference_model = load_model(
            config_name, checkpoint_path, device="cpu", decode_only=True
        )
        failures += run_fp16_combos(
            model, reference_model, cases, min_snr_db, max_snr_drop_db
        )
    else:
        if "float32" in wanted:
            click.echo(f"\n== float32 on {device}: SNR >= {MIN_SNR_FLOAT32} dB ==")
            click.echo(
                "   (float32 conv kernels round differently for different input "
                "lengths;\n    exactness is proven in float64 below)"
            )
            failures += run_combos(
                model,
                cases,
                REQUIRED_COMBOS + DIAGNOSTIC_COMBOS,
                lambda c, o, d, s: (
                    s >= MIN_SNR_FLOAT32 if (c, o) in REQUIRED_COMBOS else None
                ),
                "float32",
            )

        if "float64" in wanted:
            click.echo(
                f"\n== float64 on {device}: max|diff| <= {MAX_DIFF_FLOAT64:.0e} =="
            )
            model.to(torch.float64)
            failures += run_combos(
                model,
                cases,
                EXACT_COMBOS,
                lambda c, o, d, s: d <= MAX_DIFF_FLOAT64,
                "float64",
            )

    click.echo("\n== minimum exact overlap ==")
    if cuda or "overlap" not in wanted:
        click.echo("  skipped (needs the float64 CPU path)")
    else:
        model.to(torch.float64)
        name, indices = cases[0]
        probe = indices[..., :search_frames]
        click.echo(f"  float64, case {name}, first {probe.shape[-1]} frames, chunk 32")
        min_overlap = find_min_exact_overlap(
            model, probe, 32, max_overlap_search, floor_overlap
        )
        if min_overlap is None:
            failures.append(f"no exact overlap found up to {max_overlap_search}")
        else:
            margin = DEFAULT_OVERLAP / max(min_overlap, 1)
            click.echo(
                f"\n  minimum exact overlap: {min_overlap} frames "
                f"({min_overlap / FRAME_RATE:.2f}s of context); "
                f"default {DEFAULT_OVERLAP} gives {margin:.1f}x margin"
            )
            if margin < 2.0:
                failures.append(
                    f"default overlap {DEFAULT_OVERLAP} has only {margin:.1f}x "
                    f"margin over the measured minimum {min_overlap}; "
                    "raise the default"
                )

    click.echo("")
    if failures:
        for failure in failures:
            click.echo(f"FAIL: {failure}")
        raise SystemExit(1)
    click.echo("PASS: chunked decode reproduces single-pass decode")


if __name__ == "__main__":
    main()
