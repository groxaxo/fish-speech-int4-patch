#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8880}"
HOST="${HOST:-0.0.0.0}"
GPU_INDEX="${GPU_INDEX:-0}"
ENV_NAME="${ENV_NAME:-fish-speech-bnb4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
IDLE_TIMEOUT_SECONDS="${IDLE_TIMEOUT_SECONDS:-300}"

is_complete_checkpoint_dir() {
    local dir="$1"
    [[ -f "${dir}/codec.pth" ]] || return 1
    [[ -f "${dir}/config.json" ]] || return 1
    [[ -f "${dir}/tokenizer.json" ]] || return 1
    [[ -f "${dir}/model.safetensors.index.json" ]] && return 0
    compgen -G "${dir}/model-*.safetensors" >/dev/null
}

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    if is_complete_checkpoint_dir "${REPO_DIR}/checkpoints/s2-pro"; then
        CHECKPOINT_DIR="${REPO_DIR}/checkpoints/s2-pro"
    elif is_complete_checkpoint_dir "${REPO_DIR}/checkpoints"; then
        CHECKPOINT_DIR="${REPO_DIR}/checkpoints"
    else
        CHECKPOINT_DIR="${REPO_DIR}/checkpoints/s2-pro"
    fi
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    CONDA_BIN="/home/op/miniconda3/bin/conda"
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "Conda was not found. Run ./install_bnb4_3060.sh after installing Miniconda."
    exit 1
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python for env ${ENV_NAME} was not found at ${PYTHON_BIN}"
    echo "Run ./install_bnb4_3060.sh to create the environment first."
    exit 1
fi

if ! is_complete_checkpoint_dir "${CHECKPOINT_DIR}"; then
    echo "Missing a complete S2-Pro checkpoint at ${CHECKPOINT_DIR}"
    echo "Run ./install_bnb4_3060.sh to install dependencies and download the model."
    exit 1
fi

PORT_PIDS="$(ss -tlnp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port { if (match($0, /pid=[0-9]+/)) { pid = substr($0, RSTART + 4, RLENGTH - 4); print pid } }' | sort -u)"
if [[ -n "${PORT_PIDS}" ]]; then
    echo "Port ${PORT} is busy. Stopping PID(s): ${PORT_PIDS}"
    while read -r pid; do
        [[ -n "${pid}" ]] && kill "${pid}"
    done <<< "${PORT_PIDS}"
    sleep 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_DIR}"

echo "Launching Fish Speech S2-Pro on GPU ${GPU_INDEX}"
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "Mode: bitsandbytes NF4 4-bit, lazy load, idle shutdown after ${IDLE_TIMEOUT_SECONDS}s"

exec "${PYTHON_BIN}" tools/api_server.py \
    --llama-checkpoint-path "${CHECKPOINT_DIR}" \
    --decoder-checkpoint-path "${CHECKPOINT_DIR}/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --device cuda \
    --bnb4 \
    --half \
    --lazy-load \
    --idle-timeout-seconds "${IDLE_TIMEOUT_SECONDS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --listen "${HOST}:${PORT}"
