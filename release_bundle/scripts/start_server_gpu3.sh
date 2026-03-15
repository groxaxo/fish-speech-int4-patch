#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate fish-speech

cd /home/op/fish-speech
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro-int4-g128-mcpint4c \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --device cuda \
  --max-seq-len 4096 \
  --listen 0.0.0.0:8181
