"""Simulate a small GPU on a big one and report whether the pipeline fits.

torch.cuda.set_per_process_memory_fraction() caps PyTorch's caching allocator to
a fraction of the physical card. The CUDA context itself lives outside that cap,
so we subtract an allowance for it before computing the fraction.
"""

import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-mib", type=int, default=4096, help="simulated card size")
    p.add_argument("--context-mib", type=int, default=400, help="assumed CUDA ctx overhead")
    p.add_argument("--llama-path", default="checkpoints/s2-pro")
    p.add_argument("--codec-path", default="checkpoints/s2-pro/codec.pth")
    p.add_argument("--codec-config", default="modded_dac_vq")
    p.add_argument("--reference-id", default="beatrice10")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--bnb4", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Set near 0 to make sampling effectively greedy, which makes runs "
        "comparable across changes that alter RNG consumption.",
    )
    p.add_argument("--save-prefix", default=None, help="write generated wavs here")
    p.add_argument("--codec-decode-only", action="store_true")
    args = p.parse_args()

    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    usable = max(args.budget_mib - args.context_mib, 256)
    frac = usable / total_mib
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"card={torch.cuda.get_device_name(0)} total={total_mib:.0f} MiB")
    print(f"simulating {args.budget_mib} MiB card -> allocator cap {usable:.0f} MiB (fraction {frac:.4f})")
    print(f"config: bnb4={args.bnb4} compile={args.compile} max_seq_len={args.max_seq_len}")
    print("-" * 78, flush=True)

    from fish_speech.utils.schema import ServeTTSRequest
    from tools.server.model_manager import ModelManager

    def peak():
        return torch.cuda.max_memory_reserved() / 2**20

    t0 = time.time()
    try:
        mgr = ModelManager(
            mode="tts",
            device="cuda",
            half=True,
            compile=args.compile,
            bnb4=args.bnb4,
            llama_checkpoint_path=args.llama_path,
            decoder_checkpoint_path=args.codec_path,
            decoder_config_name=args.codec_config,
            max_seq_len=args.max_seq_len,
            lazy_load=False,
            codec_decode_only=args.codec_decode_only,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nRESULT: OOM DURING LOAD after {time.time()-t0:.0f}s")
        print(f"peak reserved before OOM: {peak():.0f} MiB")
        print(str(e).splitlines()[0])
        return 2
    print(f"\nloaded+warmed in {time.time()-t0:.0f}s; peak reserved {peak():.0f} MiB", flush=True)

    engine = mgr.tts_inference_engine
    tests = [
        ("short", "Halo, apa kabar?"),
        ("med", "Halo sayang, hari ini aku bangun pagi sekali dan langsung membuat kopi untukmu."),
        ("long", "Halo sayang, hari ini aku bangun pagi sekali dan langsung membuat kopi untukmu. "
                 "Cuacanya cerah, jadi aku pikir kita bisa jalan-jalan sore nanti kalau kamu tidak "
                 "terlalu sibuk dengan pekerjaanmu."),
    ]

    print(f"\n{'case':8s} {'wall':>7s} {'audio':>7s} {'RTF':>6s} {'peak MiB':>9s}")
    rows = []
    for label, text in tests * 2:
        req = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=args.reference_id,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=args.temperature,
            format="wav",
            seed=12345,
        )
        t = time.time()
        try:
            audio = None
            for r in engine.inference(req):
                if r.code == "final":
                    audio = r.audio
                elif r.code == "error":
                    raise r.error
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nRESULT: OOM DURING INFERENCE ({label})")
            print(f"peak reserved before OOM: {peak():.0f} MiB")
            print(str(e).splitlines()[0])
            return 2
        wall = time.time() - t
        sr, arr = audio
        dur = len(arr) / sr
        if args.save_prefix:
            import hashlib

            import soundfile as sf

            path = f"{args.save_prefix}_{label}_{len(rows)}.wav"
            sf.write(path, arr, sr)
            digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
            print(f"  saved {path} sha256={digest}")
        rows.append((label, wall, dur))
        print(f"{label:8s} {wall:7.2f} {dur:7.2f} {wall/dur:6.2f} {peak():9.0f}", flush=True)

    # least-squares fit: wall = fixed + slope * audio
    n = len(rows)
    xs = [d for _, _, d in rows]
    ys = [w for _, w, _ in rows]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    fixed = my - slope * mx

    print("-" * 78)
    print(f"fit: wall = {fixed:.2f}s + {slope:.3f} * audio   (marginal RTF {slope:.3f})")
    print(f"peak reserved: {peak():.0f} MiB of {args.budget_mib} MiB budget")
    print(f"\nRESULT: FITS in {args.budget_mib} MiB, marginal RTF {slope:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
