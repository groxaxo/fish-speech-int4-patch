#!/usr/bin/env bash
set -euo pipefail

source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate fish-speech

cd /home/op/fish-speech
python tools/rtf_benchmark.py \
  --server "${1:-http://127.0.0.1:8181}" \
  --samples "${2:-10}" \
  --warmup "${3:-2}" \
  --output "${4:-/tmp/fish_rtf_report.json}"
