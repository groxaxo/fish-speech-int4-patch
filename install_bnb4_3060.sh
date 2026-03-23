#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-fish-speech-bnb4}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_WHL_INDEX="${CUDA_WHL_INDEX:-https://download.pytorch.org/whl/cu128}"
HF_REPO="${HF_REPO:-fishaudio/s2-pro}"

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
    echo "Conda was not found. Install Miniconda or set CONDA_EXE first."
    exit 1
fi

mkdir -p "${REPO_DIR}/checkpoints"

echo "==> Preparing conda environment: ${ENV_NAME}"
if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
    echo "Environment ${ENV_NAME} already exists."
fi

cd "${REPO_DIR}"

echo "==> Installing Fish Speech with BnB NF4 support"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e ".[bnb]" --extra-index-url "${CUDA_WHL_INDEX}"

echo "==> Ensuring S2-Pro checkpoints are available locally"
if ! is_complete_checkpoint_dir "${CHECKPOINT_DIR}"; then
    HF_REPO="${HF_REPO}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" "${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO"],
    local_dir=os.environ["CHECKPOINT_DIR"],
    local_dir_use_symlinks=False,
)
PY
else
    echo "Checkpoint already present at ${CHECKPOINT_DIR}"
fi

echo
echo "Installation complete."
echo "Start the API server with:"
echo "  ./start_bnb4_3060.sh"
