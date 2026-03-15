#!/usr/bin/env python3
"""
Correlate generated audio from local fish-speech server with a Whisper-style ASR endpoint.

Usage:
  python tools/correlate_with_whisper.py \
    --server http://127.0.0.1:8181 \
    --whisper http://100.85.200.54:5092/v1/transcriptions \
    --text "Hello world" \
    --format mp3

Notes:
- The script supports multipart/form-data file upload (default). If your Whisper instance expects a different payload, adapt the --whisper-* args.
- Computes a simple WER (word error rate) between reference text and ASR result.
"""

import argparse
import requests
import tempfile
import os
import time
import json


def simple_wer(ref, hyp):
    # compute word-level Levenshtein distance / ref_length
    r = ref.strip().split()
    h = hyp.strip().split()
    n = len(r)
    if n == 0:
        return 0.0 if len(h) == 0 else 1.0
    # dp
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    edits = dp[len(r)][len(h)]
    return edits / float(n)


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
    return out, elapsed


def send_to_whisper(whisper_url, path, headers=None, payload_format='multipart'):
    if payload_format == 'multipart':
        files = {'file': open(path, 'rb')}
        r = requests.post(whisper_url, files=files, headers=headers, timeout=600)
    elif payload_format == 'json_base64':
        import base64
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        r = requests.post(whisper_url, json={'audio': b64}, headers=headers, timeout=600)
    else:
        # raw bytes
        with open(path, 'rb') as f:
            data = f.read()
        r = requests.post(whisper_url, data=data, headers=headers, timeout=600)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {'raw_text': r.text}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', default='http://127.0.0.1:8181')
    p.add_argument('--whisper', required=True, help='Full whisper URL (e.g. http://host:port/v1/transcriptions)')
    p.add_argument('--text', required=True)
    p.add_argument('--format', default='mp3')
    p.add_argument('--whisper-format', default='multipart', choices=['multipart','json_base64','raw'])
    p.add_argument('--whisper-auth', default=None, help='Optional Authorization header value (e.g. "Bearer ...")')
    p.add_argument('--out', default='/tmp/fish_whisper_correlation.json')
    args = p.parse_args()

    print('Generating audio...')
    path, gen_elapsed = generate_audio(args.server, args.text, fmt=args.format)
    print('Wrote audio to', path, 'generation time:', gen_elapsed)

    headers = {}
    if args.whisper_auth:
        headers['Authorization'] = args.whisper_auth

    print('Sending to Whisper endpoint...')
    resp = send_to_whisper(args.whisper, path, headers=headers or None, payload_format=args.whisper_format)
    # Try to extract transcript
    transcript = None
    if isinstance(resp, dict):
        for k in ('text','transcript','transcription'):
            if k in resp:
                transcript = resp[k]
                break
        if transcript is None:
            # try nested
            for v in resp.values():
                if isinstance(v, str) and len(v) > 0:
                    transcript = v
                    break
    else:
        transcript = str(resp)

    if transcript is None:
        transcript = json.dumps(resp)

    wer = simple_wer(args.text, transcript)

    report = {
        'reference': args.text,
        'transcript_raw': transcript,
        'wer': wer,
        'audio_file': path,
        'generation_time_s': gen_elapsed,
        'whisper_response': resp
    }

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    print('Wrote correlation report to', args.out)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
