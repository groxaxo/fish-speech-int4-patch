#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <whisper-url> <text> [server-url] [output-json]" >&2
  exit 1
fi

source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate fish-speech

cd /home/op/fish-speech
python tools/correlate_with_whisper.py \
  --whisper "$1" \
  --text "$2" \
  --server "${3:-http://127.0.0.1:8181}" \
  --out "${4:-/tmp/fish_whisper_correlation.json}"
