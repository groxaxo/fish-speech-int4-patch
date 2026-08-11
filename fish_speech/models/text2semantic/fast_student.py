"""Distilled student for the S2-Pro fast (depth) transformer.

The teacher fast transformer (``DualARTransformer`` in ``llama.py``) is a
10-position causal sequence model that runs once per generated frame:

    position 0    input  ``fast_project_in(slow_hidden)``  -> logits DISCARDED
                                                              (codebook 0 comes
                                                              from the slow head)
    position i>=1 input  ``fast_embeddings(code[i - 1])``   -> logits supervise
                                                              codebook i

so only positions 1..9 carry supervision. At d2560/4L the teacher costs ~425M
params of weight traffic nine times per frame, which is roughly half of decode.
This module is a narrower stand-in with the same interface:

* ``forward_parallel``      - training path, all 10 positions in one causal pass
* ``forward_generate_fast`` - inference path, one position at a time with KV
                              caches, signature-compatible with the teacher's
* ``fast_embeddings`` / ``embed`` - the (4096 -> dim) code embedding the decode
                              loop feeds back between positions

The blocks are imported from ``llama.py`` so the RMSNorm / RoPE / GQA / SwiGLU
math is literally the teacher's, only narrower. ``project_in`` (2560 -> dim)
replaces the teacher's ``fast_project_in`` (an Identity at d2560).

Unlike the teacher, ``forward_generate_fast`` accepts either a projected
``dim``-wide activation or a raw ``in_dim``-wide slow hidden and projects the
latter itself, so the decode loop only has to swap ``fast_embeddings`` and
``forward_generate_fast``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    KVCache,
    RMSNorm,
    TransformerBlock,
    precompute_freqs_cis,
)

SAVE_FORMAT = "fish-speech-fast-student"
SAVE_VERSION = 1


@dataclass
class FastStudentArgs:
    """Shape of the distilled fast transformer.

    Defaults are the first candidate from ``notes/distill-fast-student.md``:
    ~59M params (fp16 118 MB) against the teacher's ~425M.
    """

    dim: int = 1024
    n_layer: int = 4
    n_head: int = 16
    head_dim: int = 64
    n_local_heads: int = 4
    intermediate_size: int = 3072
    rope_base: float = 1000000
    norm_eps: float = 1e-6
    codebook_size: int = 4096
    num_codebooks: int = 10
    in_dim: int = 2560
    initializer_range: float = 0.02

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FastStudentArgs":
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"unknown FastStudentArgs fields: {sorted(unknown)}")
        return cls(**data)

    def block_config(self) -> BaseModelArgs:
        """The ``BaseModelArgs`` the shared ``TransformerBlock`` expects.

        Mirrors the teacher's fast block: no qkv/o bias, no qk norm, no dropout.
        """
        return BaseModelArgs(
            dim=self.dim,
            n_head=self.n_head,
            head_dim=self.head_dim,
            n_local_heads=self.n_local_heads,
            intermediate_size=self.intermediate_size,
            rope_base=self.rope_base,
            norm_eps=self.norm_eps,
            dropout=0.0,
            attention_qkv_bias=False,
            attention_o_bias=False,
            attention_qk_norm=False,
            codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            initializer_range=self.initializer_range,
            use_gradient_checkpointing=False,
        )


class FastStudent(nn.Module):
    def __init__(self, config: Optional[FastStudentArgs] = None) -> None:
        super().__init__()
        self.config = config or FastStudentArgs()
        block_config = self.config.block_config()

        self.project_in = nn.Linear(self.config.in_dim, self.config.dim)
        self.fast_embeddings = nn.Embedding(self.config.codebook_size, self.config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(block_config, use_sdpa=True)
            for _ in range(self.config.n_layer)
        )
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.output = nn.Linear(self.config.dim, self.config.codebook_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.config.num_codebooks,
                self.config.head_dim,
                self.config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    self.config.num_codebooks,
                    self.config.num_codebooks,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ init

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ----------------------------------------------------------------- utils

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _cast_in(self, x: Tensor) -> Tensor:
        """Match the caller's activations to the module dtype.

        Shards hold fp16 hiddens while the trainer keeps fp32 master weights, and
        autocast only converts inside the op - an explicit cast keeps both the
        autocast and the plain-fp32/fp16 call sites working.
        """
        return x.to(self.project_in.weight.dtype)

    def embed(self, idx: Tensor) -> Tensor:
        """Code index -> ``dim``-wide activation, the decode loop's feedback."""
        return self.fast_embeddings(idx)

    # -------------------------------------------------------------- training

    def forward_parallel(self, hiddens: Tensor, codes: Tensor) -> Tensor:
        """One causal pass over all 10 positions.

        Args:
            hiddens: ``(B, in_dim)`` (or ``(B, 1, in_dim)``) slow-model hidden.
            codes: ``(B, num_codebooks)`` sampled indices; ``codes[:, 0]`` is the
                rebased semantic index, ``codes[:, i]`` the target of position i.

        Returns:
            ``(B, num_codebooks, codebook_size)``. Position 0's logits are
            structurally meaningless (the teacher discards them); supervise
            positions 1..num_codebooks-1 only.
        """
        if hiddens.dim() == 3:
            hiddens = hiddens.reshape(hiddens.shape[0], -1)

        n = self.config.num_codebooks
        codes = codes.long()
        if codes.shape[1] != n:
            raise ValueError(f"expected {n} codes per frame, got {codes.shape[1]}")

        x = self.project_in(self._cast_in(hiddens))
        # Position i>=1 reads the code sampled at position i-1, so the last
        # codebook is an input to nothing and is dropped here.
        emb = self.fast_embeddings(codes[:, : n - 1])
        # Autocast converts the projection to bf16 but leaves the embedding
        # lookup in the parameter dtype; align them rather than letting cat
        # promote the whole sequence back to fp32.
        if emb.dtype != x.dtype:
            emb = emb.to(x.dtype)
        x = torch.cat([x[:, None], emb], dim=1)

        mask = self.causal_mask[None, None, :n, :n]
        freqs_cis = self.freqs_cis[:n]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        return self.output(self.norm(x))

    # ------------------------------------------------------------- inference

    def setup_caches(
        self,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        """Allocate the per-layer KV caches (length = num_codebooks, as teacher)."""
        if device is None:
            device = next(self.parameters()).device

        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            ).to(device)

    def clear_caches(self) -> None:
        for block in self.layers:
            block.attention.kv_cache = None

    def forward_generate_fast(
        self, x: Tensor, input_pos: Tensor, project: Optional[bool] = None
    ) -> Tensor:
        """Single position through the cached stack.

        ``x`` is ``(B, dim)``/``(B, 1, dim)`` for positions >= 1, or the raw
        ``(B, in_dim)``/``(B, 1, in_dim)`` slow hidden for position 0 - the
        latter is projected here so the teacher's call site works unchanged.

        ``project`` forces the choice; the default infers it from the width,
        which is unambiguous only while ``in_dim != dim``. A student built at
        ``in_dim == dim`` must pass it explicitly, otherwise the slow hidden
        would skip ``project_in`` here while ``forward_parallel`` still applied
        it - the exact train/serve split this module exists to avoid.
        """
        x = x.view(x.shape[0], 1, -1)
        width = x.shape[-1]

        if project is None:
            if self.config.in_dim == self.config.dim:
                raise ValueError(
                    "in_dim == dim makes the projection undecidable from the "
                    "input width; pass project=True for the slow hidden and "
                    "project=False for a code embedding"
                )
            project = width == self.config.in_dim

        if project:
            if width != self.config.in_dim:
                raise ValueError(f"expected in_dim={self.config.in_dim}, got {width}")
            x = self.project_in(self._cast_in(x))
        elif width != self.config.dim:
            raise ValueError(f"expected dim={self.config.dim}, got {width}")

        mask = self.causal_mask[None, None, input_pos, : self.config.num_codebooks]
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        return self.output(self.norm(x))

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": SAVE_FORMAT,
                "version": SAVE_VERSION,
                "args": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @staticmethod
    def load(
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> "FastStudent":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != SAVE_FORMAT:
            raise ValueError(f"{path} is not a {SAVE_FORMAT} checkpoint")

        model = FastStudent(FastStudentArgs.from_dict(payload["args"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)
        return model.eval()
