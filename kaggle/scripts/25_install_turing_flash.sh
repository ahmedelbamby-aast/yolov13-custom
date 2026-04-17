#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ "${Y13_INSTALL_TURING_FLASH:-0}" != "1" ]]; then
  echo "Skipping Turing FlashAttention install (set Y13_INSTALL_TURING_FLASH=1 to enable)."
  exit 0
fi

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" && -x "${Y13_ROOT}/.venv/Scripts/python.exe" ]]; then
  PY="${Y13_ROOT}/.venv/Scripts/python.exe"
fi
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv. Run 10_setup_uv.sh first." >&2
  exit 1
fi

WHEEL_URL_DEFAULT="https://github.com/ahmedelbamby-aast/yolov13-custom/releases/download/flash-turing-wheels-v1/flash_attn_turing-0.0.0-cp311-cp311-linux_x86_64.whl"
WHEEL_URL="${Y13_TURFLASH_WHEEL_URL:-${WHEEL_URL_DEFAULT}}"

if [[ "${Y13_TURFLASH_USE_WHEEL:-1}" == "1" ]]; then
  echo "[turflash] attempting prebuilt wheel install: ${WHEEL_URL}"
  set +e
  uv pip install --python "${PY}" -U "${WHEEL_URL}"
  rc_wheel=$?
  set -e
  if [[ "${rc_wheel}" -eq 0 ]]; then
    echo "[turflash] prebuilt wheel installed successfully."
    exit 0
  else
    echo "[turflash] prebuilt wheel install failed, falling back to source build."
  fi
fi

WORKTREE="${Y13_WORKDIR}/flash-attention-turing"
if [[ ! -d "${WORKTREE}" ]]; then
  git clone https://github.com/ssiu/flash-attention-turing "${WORKTREE}"
else
  git -C "${WORKTREE}" pull --ff-only || true
fi

git -C "${WORKTREE}" submodule update --init --recursive

export CXX=g++
export CC=gcc
export MAX_JOBS="${Y13_TURFLASH_MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${Y13_TORCH_CUDA_ARCH_LIST:-7.5}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[turflash] nvidia-smi not found; skipping Turing build on non-GPU host."
  exit 0
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi 't4'; then
  echo "[turflash] T4 GPU not detected; skipping Turing-specific build."
  exit 0
fi

if [[ -d "/usr/local/cuda" ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
if command -v nvcc >/dev/null 2>&1; then
  export CUDACXX="$(command -v nvcc)"
fi

echo "[turflash] nvcc=$(command -v nvcc || true)"
echo "[turflash] ptxas=$(command -v ptxas || true)"
nvcc --version || true
ptxas --version || true

echo "[turflash] MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

uv pip install --python "${PY}" ninja setuptools wheel

set +e
uv pip install --python "${PY}" -v --no-build-isolation --no-deps "${WORKTREE}"
rc=$?
set -e

if [[ "${rc}" -ne 0 ]]; then
  echo "[warn] Turing FlashAttention build failed; fallback backend remains active."
else
  echo "Turing FlashAttention installed successfully."
fi
