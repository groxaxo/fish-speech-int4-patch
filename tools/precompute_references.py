"""Precompute VQ codes for reference clips so the server can run decode-only.

A decode-only codec drops the encoder (~0.96 GiB of VRAM), which means it can no
longer turn reference audio into codes at request time. Run this once, with the
full codec, to write a `<clip>.tokens.pt` next to each reference clip.

    python tools/precompute_references.py
    python tools/precompute_references.py --reference-id beatrice10 --force
"""

from pathlib import Path

import click
import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.utils.file import AUDIO_EXTENSIONS, audio_to_bytes, list_files


class _Encoder(VQManager, ReferenceLoader):
    def __init__(self, decoder_model):
        ReferenceLoader.__init__(self)
        self.decoder_model = decoder_model


@click.command()
@click.option("--references-dir", default="references", type=click.Path(path_type=Path))
@click.option("--reference-id", default=None, help="only this id (default: all)")
@click.option(
    "--decoder-checkpoint-path",
    default="checkpoints/s2-pro/codec.pth",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--decoder-config-name", default="modded_dac_vq")
@click.option("--device", default="cuda")
@click.option("--force", is_flag=True, help="overwrite existing caches")
def main(
    references_dir: Path,
    reference_id: str | None,
    decoder_checkpoint_path: Path,
    decoder_config_name: str,
    device: str,
    force: bool,
):
    if not references_dir.exists():
        raise SystemExit(f"No references directory at {references_dir}")

    folders = (
        [references_dir / reference_id]
        if reference_id
        else sorted(p for p in references_dir.iterdir() if p.is_dir())
    )
    folders = [f for f in folders if f.exists()]
    if not folders:
        raise SystemExit("No reference folders found")

    # Full codec: encoding is exactly what this tool exists to do.
    decoder_model = load_decoder_model(
        config_name=decoder_config_name,
        checkpoint_path=str(decoder_checkpoint_path),
        device=device,
    )
    encoder = _Encoder(decoder_model)

    written = skipped = 0
    for folder in folders:
        for clip in list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=True):
            out = ReferenceLoader.cached_tokens_path(clip)
            if out.exists() and not force:
                logger.info(f"skip (exists): {out}")
                skipped += 1
                continue
            # Encode directly rather than via load_or_encode_reference, which
            # would return the existing cache and defeat --force.
            codes = encoder.encode_reference(
                reference_audio=audio_to_bytes(str(clip)),
                enable_reference_audio=True,
            )
            torch.save(codes.cpu(), out)
            logger.info(f"wrote {out} {tuple(codes.shape)}")
            written += 1

    logger.info(f"done: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
