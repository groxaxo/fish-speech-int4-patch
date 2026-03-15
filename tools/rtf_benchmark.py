#!/usr/bin/env python3
"""
RTF benchmark for fish-speech

Usage:
  python tools/rtf_benchmark.py --server http://127.0.0.1:8181 --samples 10 --warmup 2 --output /tmp/fish_rtf_report.json

This script POSTs text to /v1/audio/speech, measures wall time and audio duration, and computes RTF = elapsed / audio_duration.
"""

import argparse
import requests
import time
import tempfile
import os
import json
import statistics
import random

SAMPLE_TEXTS = [
    "Hello world. This is a benchmark sample for realtime factor measurement.",
    "Hola, este es un texto de ejemplo para medir el factor en tiempo real.",
    "The quick brown fox jumps over the lazy dog.",
    "Prueba en español para medir rendimiento y latencia.",
    "This is another test sentence to generate audio."
]


def duration_of_file(path):
    # try soundfile, then mutagen as fallback
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        try:
            from mutagen.mp3 import MP3
            return MP3(path).info.length
        except Exception:
            return None


def generate_audio(server, text, fmt='mp3', timeout=600):
    url = server.rstrip('/') + '/v1/audio/speech'
    payload = {"input": text, "format": fmt}
    start = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - start
    r.raise_for_status()
    ctype = r.headers.get('content-type','')
    ext = '.mp3' if 'mpeg' in ctype or 'mp3' in ctype or 'audio/mpeg' in ctype else '.wav'
    out = tempfile.mktemp(suffix=ext)
    with open(out, 'wb') as f:
        f.write(r.content)
    duration = duration_of_file(out)
    return elapsed, duration, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', default='http://127.0.0.1:8181')
    p.add_argument('--samples', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--format', default='mp3')
    p.add_argument('--output', default='/tmp/fish_rtf_report.json')
    args = p.parse_args()

    # simple readiness check
    print('Checking server readiness...')
    for _ in range(120):
        try:
            resp = requests.get(args.server.rstrip('/') + '/v1/models', timeout=5)
            if resp.status_code < 400:
                print('Server ready')
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print('Server did not respond in time; continuing but requests may fail')

    # warmup
    for i in range(args.warmup):
        text = random.choice(SAMPLE_TEXTS)
        print(f'Warmup {i+1}/{args.warmup}...')
        try:
            generate_audio(args.server, text, fmt=args.format)
        except Exception as e:
            print('Warmup failed:', e)

    results = []
    for i in range(args.samples):
        text = random.choice(SAMPLE_TEXTS)
        print(f'Run {i+1}/{args.samples}...')
        try:
            elapsed, duration, path = generate_audio(args.server, text, fmt=args.format)
        except Exception as e:
            print('Run failed:', e)
            continue
        rtf = None
        if duration and duration > 0:
            rtf = elapsed / duration
        results.append({'elapsed': elapsed, 'duration': duration, 'rtf': rtf, 'path': path, 'text': text})
        print(f'Elapsed: {elapsed:.2f}s, duration: {duration}, rtf: {rtf}')

    rtfs = [r['rtf'] for r in results if r['rtf'] is not None]
    elapsed_list = [r['elapsed'] for r in results]
    dur_list = [r['duration'] for r in results if r['duration'] is not None]
    report = {
        'samples_requested': args.samples,
        'successful_samples': len(results),
        'avg_elapsed_s': statistics.mean(elapsed_list) if elapsed_list else None,
        'median_elapsed_s': statistics.median(elapsed_list) if elapsed_list else None,
        'avg_duration_s': statistics.mean(dur_list) if dur_list else None,
        'avg_rtf': statistics.mean(rtfs) if rtfs else None,
        'median_rtf': statistics.median(rtfs) if rtfs else None,
        'runs': results
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print('Wrote report to', args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
