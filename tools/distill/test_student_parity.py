"""Parity gate for :class:`FastStudent`.

The student is trained through ``forward_parallel`` (one causal pass over the
10 fast positions) but served through ``forward_generate_fast`` (one position at
a time against a KV cache). A RoPE offset, a mask off-by-one or a cache write to
the wrong slot silently changes only the served path, so training would look
healthy and the audio would be wrong. This asserts the two paths agree.

Beyond that agreement the tests pin the structure the teacher defines
(``decode_one_token_ar`` in ``inference.py``): position ``i >= 1`` consumes the
code sampled at position ``i - 1`` and nothing later, and position 0 consumes
the slow hidden. A negative control checks the parity assertion actually has
teeth - an agreement test that cannot fail proves nothing.

Run standalone (``python tools/distill/test_student_parity.py``) or under
pytest.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fish_speech.models.text2semantic.fast_student import (  # noqa: E402
    FastStudent,
    FastStudentArgs,
)

ATOL = 2e-3
# bf16 keeps ~8 mantissa bits, so the two paths cannot agree to ATOL there; what
# has to survive is the sampled token, i.e. the argmax.
BF16_ATOL = 0.5


def _random_model(seed: int = 0, dtype: torch.dtype = torch.float32) -> FastStudent:
    torch.manual_seed(seed)
    model = FastStudent(FastStudentArgs()).to(dtype).eval()
    # The default init leaves every RMSNorm at exactly 1.0, which would hide a
    # norm applied at the wrong point; perturb them so the test has teeth.
    for name, param in model.named_parameters():
        if name.endswith("norm.weight") and param.dim() == 1:
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.05)
    return model


def _inputs(model: FastStudent, batch_size: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed + 991)
    n = model.config.num_codebooks
    hiddens = torch.randn(
        batch_size, model.config.in_dim, generator=gen, dtype=torch.float32
    )
    codes = torch.randint(
        0, model.config.codebook_size, (batch_size, n), generator=gen
    )
    # Pin the extremes of the embedding table into every run.
    codes[0, 0] = 0
    codes[0, -1] = model.config.codebook_size - 1
    return hiddens.to(model.project_in.weight.dtype), codes


@torch.no_grad()
def _sequential(model: FastStudent, hiddens, codes, freqs_shift: int = 0):
    """Serve all 10 positions one at a time through the KV caches."""
    n = model.config.num_codebooks
    model.setup_caches(hiddens.shape[0], dtype=model.project_in.weight.dtype)

    for block in model.layers:
        assert block.attention.kv_cache is not None, "setup_caches installed no cache"
        assert bool(
            (block.attention.kv_cache.k_cache == 0).all()
        ), "cache should start zeroed"

    if freqs_shift:
        # Negative control: rotate the position table so the cached path uses
        # the wrong RoPE phase. Parity must break.
        model.freqs_cis = torch.roll(model.freqs_cis, shifts=freqs_shift, dims=0)

    try:
        # Position 0 primes the cache from the slow hidden; its logits are
        # discarded by the teacher, so they are not compared.
        model.forward_generate_fast(
            hiddens, torch.tensor([0], dtype=torch.long), project=True
        )

        out = []
        for pos in range(1, n):
            x = model.embed(codes[:, pos - 1])
            out.append(
                model.forward_generate_fast(
                    x, torch.tensor([pos], dtype=torch.long), project=False
                )[:, 0]
            )
    finally:
        if freqs_shift:
            model.freqs_cis = torch.roll(model.freqs_cis, shifts=-freqs_shift, dims=0)

    touched = [
        bool((block.attention.kv_cache.k_cache[:, :, pos] != 0).any())
        for block in model.layers
        for pos in range(n)
    ]
    assert all(touched), "some KV slot was never written - the cache path is a no-op"

    model.clear_caches()
    return torch.stack(out, dim=1)


@torch.no_grad()
def run_parity(
    batch_size: int = 3,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    freqs_shift: int = 0,
):
    model = _random_model(seed, dtype)
    n = model.config.num_codebooks
    hiddens, codes = _inputs(model, batch_size, seed)

    parallel = model.forward_parallel(hiddens, codes)
    assert parallel.shape == (batch_size, n, model.config.codebook_size)

    sequential = _sequential(model, hiddens, codes, freqs_shift=freqs_shift)

    max_diff = float((parallel[:, 1:] - sequential).abs().max())
    agree = float(
        (parallel[:, 1:].argmax(-1) == sequential.argmax(-1)).float().mean()
    )
    return max_diff, agree


def test_forward_parallel_matches_incremental():
    for batch_size in (1, 3, 8):
        max_diff, agree = run_parity(batch_size=batch_size, seed=batch_size)
        assert max_diff < ATOL, (
            f"B={batch_size}: parallel vs cached decode diverge: "
            f"max |d| = {max_diff:g}"
        )
        assert agree == 1.0


def test_parity_is_sensitive_to_rope_offset():
    """Negative control: the gate must fail when the served path is wrong."""
    max_diff, _ = run_parity(freqs_shift=1)
    assert max_diff > ATOL * 10, (
        "a one-position RoPE rotation left the two paths agreeing to "
        f"{max_diff:g} - the parity assertion cannot detect a real bug"
    )


def test_bf16_argmax_survives():
    max_diff, agree = run_parity(batch_size=4, dtype=torch.bfloat16)
    # The trained student is served in low precision; the sampled token, not
    # the logit, is what has to match.
    assert agree == 1.0, f"bf16 argmax disagreement (max |d| = {max_diff:g})"
    assert max_diff < BF16_ATOL


@torch.no_grad()
def test_position_reads_previous_code_only():
    """Position i must consume code i-1 and be blind to codes >= i.

    This is the off-by-one that ``decode_one_token_ar`` fixes: the fast stack is
    fed ``fast_embeddings(code[i - 1])`` at position i, and predicts code i.
    Feeding it code i instead would leak the label and train a model that
    cannot be served.
    """
    model = _random_model(2)
    n = model.config.num_codebooks
    hiddens, codes = _inputs(model, 2, seed=2)
    base = model.forward_parallel(hiddens, codes)

    for k in range(n):
        bumped = codes.clone()
        bumped[:, k] = (bumped[:, k] + 1234) % model.config.codebook_size
        out = model.forward_parallel(hiddens, bumped)
        delta = (out - base).abs().amax(dim=(0, 2))

        # Code k enters at position k+1, so positions <= k must be untouched
        # and position k+1 (when it exists) must move.
        assert bool(
            (delta[: k + 1] == 0).all()
        ), f"code {k} leaked backwards into positions <= {k}: {delta.tolist()}"
        if k + 1 < n:
            assert float(delta[k + 1]) > 0, (
                f"position {k + 1} ignored code {k} - the input is not the "
                "previous codebook"
            )

    # And the slow hidden must reach every position.
    other = model.forward_parallel(hiddens + 1.0, codes)
    delta = (other - base).abs().amax(dim=(0, 2))
    assert bool((delta > 0).all()), f"slow hidden does not reach all positions: {delta}"


@torch.no_grad()
def test_priming_position_zero_matters():
    """Skipping the position-0 prime must change the served logits."""
    model = _random_model(3)
    n = model.config.num_codebooks
    hiddens, codes = _inputs(model, 2, seed=3)

    primed = _sequential(model, hiddens, codes)

    model.setup_caches(2, dtype=torch.float32)
    unprimed = torch.stack(
        [
            model.forward_generate_fast(
                model.embed(codes[:, pos - 1]),
                torch.tensor([pos], dtype=torch.long),
                project=False,
            )[:, 0]
            for pos in range(1, n)
        ],
        dim=1,
    )
    model.clear_caches()

    assert float((primed - unprimed).abs().max()) > ATOL * 10, (
        "dropping the position-0 prime changed nothing - attention is not "
        "reading the slow hidden through the cache"
    )


def test_project_dispatch_is_explicit_when_ambiguous():
    model = FastStudent(FastStudentArgs(in_dim=256, dim=256, n_head=4, head_dim=64))
    x = torch.randn(1, 256)
    model.setup_caches(1, dtype=torch.float32)
    try:
        model.forward_generate_fast(x, torch.tensor([0], dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError("in_dim == dim silently guessed the projection")

    model.forward_generate_fast(x, torch.tensor([0], dtype=torch.long), project=True)


def test_save_load_roundtrip(tmp_path=None):
    import tempfile

    model = _random_model(1)
    hiddens, codes = _inputs(model, 2, seed=1)
    with torch.no_grad():
        expected = model.forward_parallel(hiddens, codes)

    directory = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    path = directory / "student.pt"
    model.save(path)
    reloaded = FastStudent.load(path)
    with torch.no_grad():
        got = reloaded.forward_parallel(hiddens, codes)

    assert reloaded.config == model.config
    assert torch.equal(expected, got)


if __name__ == "__main__":
    torch.set_num_threads(4)
    model = FastStudent(FastStudentArgs())
    print(f"student params        : {model.num_parameters / 1e6:.2f}M")

    worst = 0.0
    for batch_size in (1, 3, 8):
        max_diff, agree = run_parity(batch_size=batch_size, seed=batch_size)
        worst = max(worst, max_diff)
        print(
            f"parity B={batch_size:<2d} fp32     : max |d| = {max_diff:.3e}  "
            f"argmax agree {agree:.3f}"
        )

    bf16_diff, bf16_agree = run_parity(batch_size=4, dtype=torch.bfloat16)
    print(
        f"parity B=4  bf16      : max |d| = {bf16_diff:.3e}  "
        f"argmax agree {bf16_agree:.3f}"
    )

    control, _ = run_parity(freqs_shift=1)
    print(f"negative control      : max |d| = {control:.3e}  (must be >> {ATOL:g})")

    test_position_reads_previous_code_only()
    print("code i-1 -> position i: ok (no label leak, causal)")
    test_priming_position_zero_matters()
    print("position-0 prime      : ok (slow hidden reaches the cache)")
    test_project_dispatch_is_explicit_when_ambiguous()
    print("project dispatch      : ok")
    test_save_load_roundtrip()
    print("save/load roundtrip   : bit-exact")

    ok = worst < ATOL and bf16_agree == 1.0 and control > ATOL * 10
    print(f"worst fp32 parity     : {worst:.3e}  (atol {ATOL:g})")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
