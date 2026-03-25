import shutil
from pathlib import Path

import click
import torch

from fish_speech.models.text2semantic.llama import BaseTransformer


WEIGHT_PATTERNS = [
    "model.pth",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
]


def resolve_precision(name: str) -> torch.dtype:
    precisions = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return precisions[name]


def build_output_path(checkpoint_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return checkpoint_path.parent / f"{checkpoint_path.name}-nf4"


def copy_non_weight_files(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*WEIGHT_PATTERNS),
    )


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/s2-pro",
    show_default=True,
    help="Source checkpoint directory with full-precision weights.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write the NF4-exported checkpoint into.",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
    help="CUDA device used to quantize the checkpoint.",
)
@click.option(
    "--precision",
    type=click.Choice(["fp16", "bf16"]),
    default="fp16",
    show_default=True,
    help="Compute dtype to pair with bitsandbytes NF4.",
)
@click.option(
    "--repo-id",
    type=str,
    default=None,
    help="Optional Hugging Face model repo to upload after export.",
)
@click.option(
    "--private/--public",
    default=False,
    show_default=True,
    help="Create the Hugging Face repo as private when uploading.",
)
@click.option(
    "--upload/--no-upload",
    default=False,
    show_default=True,
    help="Upload the exported folder to Hugging Face Hub.",
)
def export_nf4(
    checkpoint_path: Path,
    output_path: Path | None,
    device: str,
    precision: str,
    repo_id: str | None,
    private: bool,
    upload: bool,
) -> None:
    checkpoint_path = checkpoint_path.resolve()
    output_path = build_output_path(checkpoint_path, output_path).resolve()
    dtype = resolve_precision(precision)

    if checkpoint_path.name.endswith(("int4", "int8")) or any(
        token in checkpoint_path.name for token in ("int4", "int8")
    ):
        raise click.ClickException(
            "NF4 export expects an unquantized checkpoint directory, not a legacy int4/int8 checkpoint."
        )

    if not torch.cuda.is_available():
        raise click.ClickException(
            "CUDA is required to export a bitsandbytes NF4 checkpoint."
        )

    if not device.startswith("cuda"):
        raise click.ClickException("Use a CUDA device such as cuda:0 for NF4 export.")

    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading {checkpoint_path} with bitsandbytes NF4 on {device}...")
    model = BaseTransformer.from_pretrained(
        checkpoint_path,
        load_weights=True,
        bnb4=True,
        bnb4_compute_dtype=dtype,
    )
    model = model.to(dtype=dtype)
    model = model.to(device=device)
    model.eval()

    click.echo(f"Copying non-weight files into {output_path}...")
    copy_non_weight_files(checkpoint_path, output_path)

    click.echo("Saving quantized model.pth...")
    model.save_pretrained(output_path)

    click.echo(f"NF4 export ready at {output_path}")

    if upload:
        if repo_id is None:
            raise click.ClickException(
                "--repo-id is required when --upload is enabled."
            )

        from huggingface_hub import HfApi

        click.echo(f"Uploading {output_path} to Hugging Face repo {repo_id}...")
        api = HfApi()
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=private, exist_ok=True
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(output_path),
        )
        click.echo(f"Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    export_nf4()
