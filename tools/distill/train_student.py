"""Distill the S2-Pro fast (depth) transformer into a :class:`FastStudent`.

The teacher is the 4-layer d2560 fast stack of the serving NF4 checkpoint. Per
generated frame it is a 10-position causal sequence model:

    position 0    input  slow hidden (fast_project_in is Identity at d2560)
    position i>=1 input  fast_embeddings(code[i - 1])   -> predicts code i

so only positions 1..9 carry supervision (codebook 0 comes from the slow head).
Shards under ``notes/distill/shards`` hold exactly the ``(hidden, codes)`` pairs
the sampler consumed, which is enough to recompute the teacher's full 4096-way
distribution on the fly - no logits are stored.

Three things here deserve a note.

**Teacher loading never touches ``from_pretrained``.** Materialising the 36 slow
layers OOM-kills the 16.7 GB WSL VM. Instead the checkpoint is mmap'd and only
the ``fast_*`` tensors are pulled out, dequantised from NF4, and cached as a
~850 MB fp16 module at ``notes/distill/teacher_fast_fp16.pt``.

**The step is gradient-accumulated.** ``--batch-size`` is the optimiser batch;
it is processed in ``--micro-batch`` slices so peak VRAM is set by the slice,
not by the batch. The loss is identical to the un-chunked one (each slice is
weighted by its share of the batch).

**Everything is resumable.** Model, optimiser, scheduler, step, RNG and the
data-split hash all live in the checkpoint; ``--resume`` picks up ``latest.pt``.

Usage::

    python tools/distill/train_student.py --steps 6000 --batch-size 2048
    python tools/distill/train_student.py --resume
    python tools/distill/train_student.py --teacher-only   # build/validate cache
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fish_speech.models.text2semantic.fast_student import (  # noqa: E402
    FastStudent,
    FastStudentArgs,
)
from fish_speech.models.text2semantic.llama import (  # noqa: E402
    BaseModelArgs,
    RMSNorm,
    TransformerBlock,
    precompute_freqs_cis,
)

DEFAULT_CHECKPOINT = Path("checkpoints/s2-pro")
DEFAULT_SHARDS = Path("notes/distill/shards")
DEFAULT_WORKDIR = Path("notes/distill")

TEACHER_CACHE_VERSION = 1


# --------------------------------------------------------------------- teacher


class TeacherFast(nn.Module):
    """The teacher's fast stack, dequantised, in plain fp16.

    Structure mirrors ``DualARTransformer``'s fast half exactly (same
    ``TransformerBlock``), so the parameter names are the checkpoint's and
    ``load_state_dict`` is a straight copy. ``fast_project_in`` is an Identity
    at ``fast_dim == dim`` and therefore has no weights to carry.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = config
        block = BaseModelArgs(
            dim=config["fast_dim"],
            n_head=config["fast_n_head"],
            head_dim=config["fast_head_dim"],
            n_local_heads=config["fast_n_local_heads"],
            intermediate_size=config["fast_intermediate_size"],
            rope_base=config["rope_base"],
            norm_eps=config["norm_eps"],
            dropout=0.0,
            attention_qkv_bias=config["fast_attention_qkv_bias"],
            attention_o_bias=config["fast_attention_o_bias"],
            attention_qk_norm=config["fast_attention_qk_norm"],
            codebook_size=config["codebook_size"],
            num_codebooks=config["num_codebooks"],
            use_gradient_checkpointing=False,
        )
        self.num_codebooks = config["num_codebooks"]
        self.dim = config["fast_dim"]
        self.codebook_size = config["codebook_size"]

        self.fast_embeddings = nn.Embedding(self.codebook_size, self.dim)
        self.fast_layers = nn.ModuleList(
            TransformerBlock(block, use_sdpa=True) for _ in range(config["n_fast_layer"])
        )
        self.fast_norm = RMSNorm(self.dim, eps=config["norm_eps"])
        self.fast_output = nn.Linear(self.dim, self.codebook_size, bias=False)

        self.register_buffer(
            "fast_freqs_cis",
            precompute_freqs_cis(
                self.num_codebooks, config["fast_head_dim"], config["rope_base"]
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(self.num_codebooks, self.num_codebooks, dtype=torch.bool)
            ),
            persistent=False,
        )

    def forward_parallel(self, hiddens: torch.Tensor, codes: torch.Tensor):
        """``(B, 2560)`` + ``(B, 10)`` -> ``(B, 10, 4096)``.

        Position 0's logits are structurally meaningless (the teacher discards
        them at decode); supervise 1..9.
        """
        n = self.num_codebooks
        x = hiddens.to(self.fast_output.weight.dtype)
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        # Position i>=1 reads the code sampled at position i-1, so the last
        # codebook is an input to nothing.
        x = torch.cat([x[:, None], self.fast_embeddings(codes[:, : n - 1])], dim=1)

        mask = self.causal_mask[None, None, :n, :n]
        freqs_cis = self.fast_freqs_cis[:n]
        for layer in self.fast_layers:
            x = layer(x, freqs_cis, mask)
        return self.fast_output(self.fast_norm(x))


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _dequantize_fast_tensors(state_dict, device: torch.device):
    """Pull the ``fast_*`` weights out of the NF4 checkpoint as plain tensors.

    The groxaxo release stores every large linear as a bitsandbytes NF4 blob (a
    flat uint8 tensor plus ``<name>.absmax`` / ``.quant_map`` / ``.nested_*`` /
    ``.quant_state.bitsandbytes__nf4`` companions) while the small tensors -
    embeddings and RMSNorm gains - stay plain fp16. Both cases appear here.
    """
    import bitsandbytes.functional as BF

    fast_keys = [k for k in state_dict if k.startswith("fast_")]
    if any(k.startswith("fast_project_in") for k in fast_keys):
        raise RuntimeError(
            "checkpoint carries fast_project_in weights; this trainer assumes "
            "the Identity projection of fast_dim == dim"
        )

    quant_suffixes = (
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state.bitsandbytes__nf4",
        ".quant_state.bitsandbytes__fp4",
    )
    weight_keys = [k for k in fast_keys if not k.endswith(quant_suffixes)]

    out: dict[str, torch.Tensor] = {}
    nf4_blocksize: dict[str, int] = {}
    for key in sorted(weight_keys):
        raw = state_dict[key]
        companions = {
            k[len(key) + 1 :]: state_dict[k]
            for k in fast_keys
            if k.startswith(key + ".")
        }
        if not companions:
            weight = raw.to(device=device, dtype=torch.float16)
            kind = "plain"
        else:
            qs = BF.QuantState.from_dict(qs_dict=dict(companions), device=device)
            weight = BF.dequantize_4bit(raw.to(device), qs).to(torch.float16)
            nf4_blocksize[key] = qs.blocksize
            kind = f"nf4/{qs.blocksize}"
        out[key] = weight

        w = weight.float()
        amax, std = float(w.abs().max()), float(w.std())
        logger.info(
            f"  {key:<50s} {kind:<9s} {str(tuple(weight.shape)):<15s} "
            f"max {amax:7.3f} std {std:.4f}"
        )
        if not (math.isfinite(amax) and math.isfinite(std)):
            raise RuntimeError(f"{key}: non-finite weights after dequantisation")
        if amax == 0.0 or std == 0.0:
            raise RuntimeError(f"{key}: degenerate (all-zero) weights")
        if amax > 100.0:
            raise RuntimeError(f"{key}: implausible magnitude {amax}")

    if not nf4_blocksize:
        raise RuntimeError("no NF4 weights found; is this the prequantised release?")

    return out, nf4_blocksize


def _validate_dequant_roundtrip(
    weights: dict, nf4_blocksize: dict, device: torch.device
) -> float:
    """Re-quantise each NF4-derived weight and measure how much it moves.

    A tensor that already came off an NF4 grid lands back on that same grid, so
    cosine ~= 1. A wrong ``quant_state`` - blocksize, absmax pairing, nesting -
    shows up here as a cosine well below 1 even though the tensor still looks
    finite and sane. Only the genuinely NF4 keys are checked: the plain fp16
    embeddings and norm gains were never on a 4-bit grid and would fail by
    construction.
    """
    import bitsandbytes.functional as BF

    worst = 1.0
    for key, blocksize in sorted(nf4_blocksize.items()):
        w = weights[key]
        packed, qs = BF.quantize_4bit(
            w, blocksize=blocksize, quant_type="nf4", compress_statistics=True
        )
        back = BF.dequantize_4bit(packed, qs).to(torch.float32)
        cos = float(
            F.cosine_similarity(back.reshape(1, -1), w.float().reshape(1, -1)).item()
        )
        logger.info(f"  requant cosine  {key:<50s} {cos:.6f}")
        worst = min(worst, cos)
        del packed, back
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return worst


def _empty_teacher(config: dict, device: torch.device) -> TeacherFast:
    """Allocate the teacher straight onto the target device in fp16.

    Building it the usual way would first materialise 425M fp32 parameters in
    host RAM, and host RAM is the scarce resource here (16.7 GB WSL VM).
    """
    with torch.device(device), _default_dtype(torch.float16):
        return TeacherFast(config)


def build_teacher(
    checkpoint_dir: Path,
    cache_path: Path,
    device: torch.device,
    refresh: bool = False,
) -> TeacherFast:
    config = json.loads((checkpoint_dir / "config.json").read_text())
    if config["fast_dim"] != config["dim"]:
        raise RuntimeError(
            f"fast_dim {config['fast_dim']} != dim {config['dim']}: the shards "
            "hold un-projected slow hiddens, so fast_project_in must be Identity"
        )

    if cache_path.exists() and not refresh:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("version") == TEACHER_CACHE_VERSION:
            logger.info(f"teacher: loading cached fp16 stack from {cache_path}")
            teacher = _empty_teacher(payload["config"], device)
            teacher.load_state_dict(payload["state_dict"], strict=True)
            teacher.eval().requires_grad_(False)
            return teacher
        logger.warning(f"teacher cache {cache_path} is stale; rebuilding")

    logger.info(f"teacher: mmap-loading {checkpoint_dir / 'model.pth'} (fast_* only)")
    state_dict = torch.load(
        checkpoint_dir / "model.pth", map_location="cpu", mmap=True, weights_only=True
    )
    try:
        weights, nf4_blocksize = _dequantize_fast_tensors(state_dict, device)
    finally:
        # Drop the mmap handle before anything else allocates.
        del state_dict

    logger.info("teacher: dequant round-trip check")
    worst_cos = _validate_dequant_roundtrip(weights, nf4_blocksize, device)
    if worst_cos < 0.999:
        raise RuntimeError(
            f"NF4 re-quantise round-trip cosine {worst_cos:.6f} < 0.999 - the "
            "quant_state pairing is probably wrong"
        )
    logger.info(f"teacher: worst re-quantise cosine {worst_cos:.6f}")

    teacher = _empty_teacher(config, device)
    teacher.load_state_dict(weights, strict=True)
    teacher.eval().requires_grad_(False)
    del weights

    _validate_teacher_forward(teacher, device)

    params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"teacher: {params / 1e6:.1f}M params, {params * 2 / 1e6:.0f} MB fp16")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "version": TEACHER_CACHE_VERSION,
            "config": config,
            "state_dict": {k: v.cpu() for k, v in teacher.state_dict().items()},
        },
        tmp,
    )
    os.replace(tmp, cache_path)
    logger.info(f"teacher: cached to {cache_path}")
    return teacher


@torch.no_grad()
def _validate_teacher_forward(teacher: TeacherFast, device: torch.device) -> None:
    """Sanity-check the assembled stack on a probe batch.

    Also compares the fused SDPA attention used here against the teacher's own
    ``eq_scaled_dot_product_attention`` (the serving path builds fast layers
    with ``use_sdpa=False``); they must agree or the distillation target is not
    the model that serves.
    """
    n, b = teacher.num_codebooks, 8
    hiddens = torch.randn(b, teacher.dim, device=device, dtype=torch.float16)
    codes = torch.randint(0, 1024, (b, n), device=device)

    logits = teacher.forward_parallel(hiddens, codes)
    if not torch.isfinite(logits).all():
        raise RuntimeError("teacher produced non-finite logits")
    if logits.shape != (b, n, teacher.codebook_size):
        raise RuntimeError(f"teacher logit shape {tuple(logits.shape)} unexpected")

    for layer in teacher.fast_layers:
        layer.attention.use_sdpa = False
    eq = teacher.forward_parallel(hiddens, codes)
    for layer in teacher.fast_layers:
        layer.attention.use_sdpa = True

    diff = float((logits.float() - eq.float()).abs().max())
    scale = float(logits.float().abs().max())
    agree = float((logits.argmax(-1) == eq.argmax(-1)).float().mean())
    logger.info(
        f"teacher: probe logits |max| {scale:.2f}, sdpa vs eq-attention "
        f"max |d| {diff:.3e} (fp16 accumulation), argmax agree {agree:.4f}"
    )
    if diff > 0.05 * max(scale, 1.0) or agree < 0.99:
        raise RuntimeError(
            f"sdpa and eq attention disagree: max |d| {diff:.3e}, agree {agree:.4f}"
        )


@torch.no_grad()
def validate_teacher_alignment(
    teacher: TeacherFast, data: ShardData, device: torch.device, frames: int = 512
) -> dict:
    """Score the reassembled teacher against the codes it actually sampled.

    This is the only check that can catch a wrong ``(position, codebook)``
    pairing, because every self-consistency test agrees with itself no matter
    which way the sequence is wired. If position i really predicts codebook i
    from ``fast_embeddings(code[i-1])``, the cross-entropy of the captured codes
    under this stack has to be low - the shards *are* samples from it. Both
    off-by-one wirings are scored alongside as controls; either beating the
    intended one means the loss would be training on the wrong target.
    """
    idx = data.val_idx[: min(frames, data.val_idx.numel())]
    hiddens = data.hiddens[idx].to(device)
    codes = data.codes[idx].to(device).long()
    logits = teacher.forward_parallel(hiddens, codes).float()

    def ce(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(
            F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
        )

    aligned = ce(logits[:, 1:], codes[:, 1:])
    shift_position = ce(logits[:, :-1], codes[:, 1:])
    shift_code = ce(logits[:, 1:], codes[:, :-1])
    uniform = math.log(teacher.codebook_size)
    top1 = float((logits[:, 1:].argmax(-1) == codes[:, 1:]).float().mean())

    logger.info(
        f"teacher alignment on {idx.numel()} captured frames: "
        f"CE(aligned) {aligned:.3f} vs CE(position-shifted) {shift_position:.3f} "
        f"vs CE(code-shifted) {shift_code:.3f} vs uniform {uniform:.3f}; "
        f"argmax hits the sampled code {top1:.1%} of the time"
    )
    if not (aligned < shift_position and aligned < shift_code):
        raise RuntimeError(
            "a shifted wiring explains the captured codes better than the "
            "intended one - position/codebook alignment is wrong"
        )
    if aligned > 0.75 * uniform:
        raise RuntimeError(
            f"teacher CE {aligned:.3f} on its own samples is near uniform "
            f"({uniform:.3f}); the dequantised stack is not the sampler"
        )
    return {
        "event": "teacher_alignment",
        "ce_aligned": round(aligned, 4),
        "ce_position_shifted": round(shift_position, 4),
        "ce_code_shifted": round(shift_code, 4),
        "ce_uniform": round(uniform, 4),
        "argmax_hits_sample": round(top1, 4),
    }


# ------------------------------------------------------------------------ data


@dataclass
class ShardData:
    hiddens: torch.Tensor  # (N, 2560) fp16, host RAM
    codes: torch.Tensor  # (N, 10) int16, host RAM
    train_idx: torch.Tensor  # (Ntrain,) int64
    val_idx: torch.Tensor  # (Nval,) int64
    split_hash: str
    n_utts: int

    @property
    def n_frames(self) -> int:
        return self.hiddens.shape[0]


def _utt_is_val(utt_id: str, val_frac: float) -> bool:
    """Deterministic per-utterance split.

    Hashing the id (not the position) keeps every already-assigned utterance on
    the same side of the split when the shard set grows, which it will: a ~2500
    utterance capture is still running.
    """
    digest = hashlib.blake2b(utt_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < val_frac


def load_shards(
    shard_dir: Path,
    cache_path: Path,
    val_frac: float = 0.02,
    refresh: bool = False,
    num_codebooks: int = 10,
    in_dim: int = 2560,
) -> ShardData:
    paths = sorted(shard_dir.glob("*.pt"))
    if not paths:
        raise click.ClickException(f"no shards under {shard_dir}")
    fingerprint = hashlib.blake2b(
        "\n".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode(),
        digest_size=16,
    ).hexdigest()

    payload = None
    if cache_path.exists() and not refresh:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") != fingerprint:
            logger.info(
                f"data cache is for a different shard set "
                f"({payload.get('n_utts')} utts) - rebuilding for {len(paths)}"
            )
            payload = None

    if payload is None:
        t0 = time.time()
        hidden_parts, code_parts, utt_ids, lengths = [], [], [], []
        for path in paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            h, c = shard["hiddens"], shard["codes"]
            if h.dim() != 2 or h.shape[1] != in_dim:
                raise click.ClickException(f"{path}: hidden shape {tuple(h.shape)}")
            if c.shape != (h.shape[0], num_codebooks):
                raise click.ClickException(f"{path}: code shape {tuple(c.shape)}")
            hidden_parts.append(h.to(torch.float16))
            code_parts.append(c.to(torch.int16))
            utt_ids.append(str(shard.get("meta", {}).get("utt_id", path.stem)))
            lengths.append(h.shape[0])

        hiddens = torch.cat(hidden_parts)
        codes = torch.cat(code_parts)
        del hidden_parts, code_parts
        logger.info(
            f"data: read {len(paths)} shards / {hiddens.shape[0]} frames in "
            f"{time.time() - t0:.1f}s"
        )
        payload = {
            "fingerprint": fingerprint,
            "hiddens": hiddens,
            "codes": codes,
            "utt_ids": utt_ids,
            "lengths": torch.tensor(lengths, dtype=torch.int64),
            "n_utts": len(paths),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, cache_path)
        logger.info(f"data: cached to {cache_path}")

    hiddens, codes = payload["hiddens"], payload["codes"]
    if not torch.isfinite(hiddens.float()).all():
        raise click.ClickException("shards contain non-finite hidden states")
    lo, hi = int(codes.min()), int(codes.max())
    if lo < 0 or hi >= 4096:
        raise click.ClickException(f"code index out of range: [{lo}, {hi}]")

    train_chunks, val_chunks, val_utts = [], [], []
    offset = 0
    for utt_id, length in zip(payload["utt_ids"], payload["lengths"].tolist()):
        idx = torch.arange(offset, offset + length, dtype=torch.int64)
        if _utt_is_val(utt_id, val_frac):
            val_chunks.append(idx)
            val_utts.append(utt_id)
        else:
            train_chunks.append(idx)
        offset += length

    if not val_chunks:
        # Tiny shard sets can hash entirely into train; hold out the last
        # utterance so the eval path still runs.
        val_chunks.append(train_chunks.pop())
        val_utts.append(payload["utt_ids"][-1])
        logger.warning("no utterance hashed into val; holding out one by position")

    train_idx = torch.cat(train_chunks)
    val_idx = torch.cat(val_chunks)
    split_hash = hashlib.blake2b(
        "\n".join(sorted(val_utts)).encode(), digest_size=8
    ).hexdigest()

    logger.info(
        f"data: {hiddens.shape[0]} frames, {payload['n_utts']} utts, "
        f"train {train_idx.numel()} / val {val_idx.numel()} "
        f"({len(val_utts)} utts), split {split_hash}"
    )
    return ShardData(
        hiddens=hiddens,
        codes=codes,
        train_idx=train_idx,
        val_idx=val_idx,
        split_hash=split_hash,
        n_utts=payload["n_utts"],
    )


# ------------------------------------------------------------------------ loss


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logp: torch.Tensor,
    teacher_p: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float,
):
    """KL(teacher || student) over positions 1..9 plus a CE anchor.

    All three tensors are already sliced to positions 1..9. KL is computed in
    fp32 regardless of the autocast dtype: the 4096-way distribution is peaky
    and bf16 log-probs lose the tail the student is meant to learn.
    """
    log_q = torch.log_softmax(student_logits.float(), dim=-1)
    kl = (teacher_p * (teacher_logp - log_q)).sum(-1).mean()
    ce = F.nll_loss(log_q.reshape(-1, log_q.shape[-1]), targets.reshape(-1))
    return kl + ce_weight * ce, kl.detach(), ce.detach()


@torch.no_grad()
def teacher_targets(teacher: TeacherFast, hiddens, codes):
    logits = teacher.forward_parallel(hiddens, codes)[:, 1:].float()
    log_p = torch.log_softmax(logits, dim=-1)
    del logits
    return log_p, log_p.exp()


# ------------------------------------------------------------------------ eval


@torch.no_grad()
def evaluate(
    student: FastStudent,
    teacher: TeacherFast,
    data: ShardData,
    device: torch.device,
    frames: int,
    micro: int,
    seed: int = 1234,
) -> dict:
    """Val KL plus per-codebook top-1 agreement with the teacher's argmax."""
    student.eval()
    gen = torch.Generator().manual_seed(seed)
    pool = data.val_idx
    if pool.numel() > frames:
        pick = torch.randperm(pool.numel(), generator=gen)[:frames]
        pool = pool[pick]

    n_pos = student.config.num_codebooks - 1
    kl_sum = torch.zeros((), device=device, dtype=torch.float64)
    ce_sum = torch.zeros((), device=device, dtype=torch.float64)
    agree = torch.zeros(n_pos, device=device, dtype=torch.float64)
    seen = 0

    for start in range(0, pool.numel(), micro):
        idx = pool[start : start + micro]
        hiddens = data.hiddens[idx].to(device, non_blocking=True)
        codes = data.codes[idx].to(device, non_blocking=True).long()

        t_logits = teacher.forward_parallel(hiddens, codes)[:, 1:].float()
        log_p = torch.log_softmax(t_logits, dim=-1)
        p = log_p.exp()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            s_logits = student.forward_parallel(hiddens, codes)[:, 1:]
        log_q = torch.log_softmax(s_logits.float(), dim=-1)

        kl_sum += (p * (log_p - log_q)).sum(-1).mean(-1).sum().double()
        ce_sum += (
            F.nll_loss(
                log_q.reshape(-1, log_q.shape[-1]),
                codes[:, 1:].reshape(-1),
                reduction="none",
            )
            .view(idx.numel(), n_pos)
            .mean(-1)
            .sum()
            .double()
        )
        agree += (t_logits.argmax(-1) == log_q.argmax(-1)).double().sum(0)
        seen += idx.numel()

    student.train()
    per_cb = (agree / max(seen, 1)).tolist()
    return {
        "val_kl": float(kl_sum / max(seen, 1)),
        "val_ce": float(ce_sum / max(seen, 1)),
        "val_frames": seen,
        # index j of per_cb is position j+1, i.e. codebook j+1
        "agree": {f"cb{j + 1}": round(v, 4) for j, v in enumerate(per_cb)},
        "agree_mean": round(sum(per_cb) / len(per_cb), 4),
    }


# --------------------------------------------------------------------- helpers


def lr_lambda_factory(warmup: int, total: int, lr: float, final_lr: float):
    floor = final_lr / lr

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def _chunks(idx: torch.Tensor, size: int) -> Iterable[torch.Tensor]:
    for start in range(0, idx.numel(), size):
        yield idx[start : start + size]


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------------- cli


@click.command()
@click.option("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, show_default=True)
@click.option("--shards", type=Path, default=DEFAULT_SHARDS, show_default=True)
@click.option("--workdir", type=Path, default=DEFAULT_WORKDIR, show_default=True)
@click.option("--steps", type=int, default=6000, show_default=True)
@click.option("--batch-size", type=int, default=2048, show_default=True,
              help="Optimiser batch in frames; split into --micro-batch slices.")
@click.option("--micro-batch", type=int, default=512, show_default=True,
              help="Frames per forward/backward slice; sets peak VRAM.")
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--final-lr", type=float, default=3e-5, show_default=True)
@click.option("--warmup", type=int, default=100, show_default=True)
@click.option("--weight-decay", type=float, default=0.01, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--ce-weight", type=float, default=0.1, show_default=True)
@click.option("--eval-every", type=int, default=250, show_default=True)
@click.option("--eval-frames", type=int, default=4096, show_default=True)
@click.option("--save-every", type=int, default=500, show_default=True)
@click.option("--val-frac", type=float, default=0.02, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--dim", type=int, default=1024, show_default=True)
@click.option("--n-layer", type=int, default=4, show_default=True)
@click.option("--n-head", type=int, default=16, show_default=True)
@click.option("--head-dim", type=int, default=64, show_default=True)
@click.option("--n-local-heads", type=int, default=4, show_default=True)
@click.option("--intermediate-size", type=int, default=3072, show_default=True)
@click.option("--vram-budget-gb", type=float, default=4.0, show_default=True,
              help="Halve --micro-batch and warn if step 1 peaks above this.")
@click.option("--resume", is_flag=True, help="Continue from ckpt/latest.pt.")
@click.option("--refresh-data", is_flag=True, help="Re-read the shard directory.")
@click.option("--refresh-teacher", is_flag=True, help="Re-derive the fp16 teacher.")
@click.option("--teacher-only", is_flag=True, help="Build/validate the teacher and exit.")
@click.option("--device", type=str, default="cuda")
def main(**opt):
    workdir: Path = opt["workdir"]
    device = torch.device(opt["device"] if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("no CUDA device - running on CPU, this will be very slow")

    torch.manual_seed(opt["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    teacher = build_teacher(
        opt["checkpoint"],
        workdir / "teacher_fast_fp16.pt",
        device,
        refresh=opt["refresh_teacher"],
    )
    data = load_shards(
        opt["shards"],
        workdir / "data_cache.pt",
        val_frac=opt["val_frac"],
        refresh=opt["refresh_data"],
        num_codebooks=teacher.num_codebooks,
        in_dim=teacher.dim,
    )

    alignment = validate_teacher_alignment(teacher, data, device)
    _append_jsonl(workdir / "train_log.jsonl", alignment)

    if opt["teacher_only"]:
        logger.info("teacher built and validated; exiting (--teacher-only)")
        return

    args = FastStudentArgs(
        dim=opt["dim"],
        n_layer=opt["n_layer"],
        n_head=opt["n_head"],
        head_dim=opt["head_dim"],
        n_local_heads=opt["n_local_heads"],
        intermediate_size=opt["intermediate_size"],
        rope_base=teacher.cfg["rope_base"],
        norm_eps=teacher.cfg["norm_eps"],
        codebook_size=teacher.codebook_size,
        num_codebooks=teacher.num_codebooks,
        in_dim=teacher.dim,
    )

    ckpt_dir = workdir / "ckpt"
    latest = ckpt_dir / "latest.pt"
    state = None
    if opt["resume"]:
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            args = FastStudentArgs.from_dict(state["student_args"])
            logger.info(f"resuming from {latest} at step {state['step']}")
        else:
            logger.warning(f"--resume but {latest} does not exist; starting fresh")

    student = FastStudent(args).to(device)
    logger.info(f"student: {student.num_parameters / 1e6:.2f}M params")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=opt["lr"],
        betas=(0.9, 0.95),
        weight_decay=opt["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda_factory(opt["warmup"], opt["steps"], opt["lr"], opt["final_lr"]),
    )

    sampler = torch.Generator()
    sampler.manual_seed(opt["seed"] + 7)
    start_step, best_val = 0, float("inf")
    micro = opt["micro_batch"]

    if state is not None:
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start_step = state["step"]
        best_val = state.get("best_val", float("inf"))
        micro = state.get("micro_batch", micro)
        if state.get("split_hash") != data.split_hash:
            logger.warning(
                f"data split changed since the checkpoint "
                f"({state.get('split_hash')} -> {data.split_hash}); new shards "
                "landed in val, so val numbers are not comparable across the "
                "break (train frames are still unseen-safe: the split is by "
                "utterance hash)"
            )
        del state

    student.train()
    batch = opt["batch_size"]
    micro = min(micro, batch)
    log_path = workdir / "train_log.jsonl"

    logger.info(
        f"train: {start_step} -> {opt['steps']} steps, batch {batch} "
        f"({math.ceil(batch / micro)} x {micro}), device {device}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    t_window = t_start
    window_steps = 0

    for step in range(start_step, opt["steps"]):
        optimizer.zero_grad(set_to_none=True)
        pick = torch.randint(
            0, data.train_idx.numel(), (batch,), generator=sampler, dtype=torch.int64
        )
        idx = data.train_idx[pick]

        loss_sum = kl_sum = ce_sum = 0.0
        for slice_idx in _chunks(idx, micro):
            share = slice_idx.numel() / batch
            hiddens = data.hiddens[slice_idx].to(device, non_blocking=True)
            codes = data.codes[slice_idx].to(device, non_blocking=True).long()

            log_p, p = teacher_targets(teacher, hiddens, codes)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                s_logits = student.forward_parallel(hiddens, codes)[:, 1:]
            loss, kl, ce = distill_loss(
                s_logits, log_p, p, codes[:, 1:], opt["ce_weight"]
            )
            (loss * share).backward()

            loss_sum += float(loss.detach()) * share
            kl_sum += float(kl) * share
            ce_sum += float(ce) * share
            del log_p, p, s_logits, loss, kl, ce

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(student.parameters(), opt["grad_clip"])
        )
        optimizer.step()
        scheduler.step()
        window_steps += 1

        if step == start_step and device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            logger.info(f"step {step}: peak VRAM {peak:.2f} GiB at micro {micro}")
            while peak > opt["vram_budget_gb"] and micro > 32:
                micro //= 2
                logger.warning(
                    f"peak VRAM {peak:.2f} GiB exceeds the "
                    f"{opt['vram_budget_gb']:.1f} GiB budget - halving micro-batch "
                    f"to {micro} (the optimiser batch is unchanged)"
                )
                # One slice is roughly linear in memory; re-measure next step.
                peak /= 2
            torch.cuda.reset_peak_memory_stats(device)

        if step % 25 == 0 or step == opt["steps"] - 1:
            elapsed = time.time() - t_window
            sps = window_steps / max(elapsed, 1e-9)
            t_window, window_steps = time.time(), 0
            record = {
                "step": step,
                "loss": round(loss_sum, 5),
                "kl": round(kl_sum, 5),
                "ce": round(ce_sum, 5),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": round(grad_norm, 4),
                "steps_per_s": round(sps, 3),
                "micro_batch": micro,
            }
            if device.type == "cuda":
                record["vram_peak_gib"] = round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 3
                )
            logger.info(
                f"step {step:5d} loss {loss_sum:8.4f} kl {kl_sum:8.4f} "
                f"ce {ce_sum:8.4f} lr {record['lr']:.2e} "
                f"gn {grad_norm:6.3f} {sps:.2f} it/s"
            )
            _append_jsonl(log_path, record)

        do_eval = (step + 1) % opt["eval_every"] == 0 or step == opt["steps"] - 1
        do_save = (step + 1) % opt["save_every"] == 0 or step == opt["steps"] - 1

        metrics = None
        if do_eval:
            metrics = evaluate(
                student, teacher, data, device, opt["eval_frames"], micro
            )
            metrics.update({"step": step, "event": "eval"})
            logger.info(
                f"eval  {step:5d} val_kl {metrics['val_kl']:.4f} "
                f"val_ce {metrics['val_ce']:.4f} | "
                f"cb1 {metrics['agree']['cb1']:.3f} cb2 {metrics['agree']['cb2']:.3f} "
                f"cb5 {metrics['agree']['cb5']:.3f} cb9 {metrics['agree']['cb9']:.3f} "
                f"| mean {metrics['agree_mean']:.3f}"
            )
            _append_jsonl(log_path, metrics)
            t_window = time.time()

        if do_save or (metrics is not None and metrics["val_kl"] < best_val):
            payload = {
                "step": step + 1,
                "student_args": student.config.to_dict(),
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "sampler_rng": sampler.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "split_hash": data.split_hash,
                "micro_batch": micro,
                "best_val": best_val,
                "metrics": metrics,
            }
            if metrics is not None and metrics["val_kl"] < best_val:
                best_val = metrics["val_kl"]
                payload["best_val"] = best_val
                _atomic_save(payload, ckpt_dir / "best.pt")
                student.save(ckpt_dir / "best_student.pt")
                logger.info(f"saved best (val_kl {best_val:.4f})")
            if do_save:
                _atomic_save(payload, latest)
                logger.info(f"saved {latest} at step {step + 1}")

    student.save(ckpt_dir / "student_final.pt")
    total = time.time() - t_start
    done = opt["steps"] - start_step
    logger.info(
        f"done: {done} steps in {total / 60:.1f} min "
        f"({total / max(done, 1):.3f} s/step); final student at "
        f"{ckpt_dir / 'student_final.pt'}"
    )


if __name__ == "__main__":
    main()
