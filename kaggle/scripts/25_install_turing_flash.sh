#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ "${Y13_INSTALL_TURING_FLASH:-0}" != "1" ]]; then
  echo "Skipping Turing FlashAttention install (set Y13_INSTALL_TURING_FLASH=1 to enable)."
  exit 0
fi

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv. Run 10_setup_uv.sh first." >&2
  exit 1
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

# Ensure CUDA compiler toolchain/headers match torch CUDA major (cu13 wheels) when system CUDA is older.
uv pip install --python "${PY}" \
  "nvidia-cuda-nvcc==${Y13_NVCC_VERSION:-13.0.88}" \
  "nvidia-cuda-crt==${Y13_CUDA_CRT_VERSION:-13.2.51}" \
  "nvidia-nvvm==${Y13_NVVM_VERSION:-13.2.51}" \
  "nvidia-cuda-cccl==${Y13_CUDA_CCCL_VERSION:-13.0.85}"
for candidate in "${Y13_VENV}/lib/python"*/site-packages/nvidia/cu13/bin; do
  if [[ -d "${candidate}" ]]; then
    export PATH="${candidate}:${PATH}"
    export LD_LIBRARY_PATH="${candidate}:${LD_LIBRARY_PATH:-}"
    export CUDACXX="${candidate}/nvcc"
    break
  fi
done
for candidate in "${Y13_VENV}/lib/python"*/site-packages/nvidia/cu13; do
  if [[ -d "${candidate}" ]]; then
    export CUDA_HOME="${candidate}"
    break
  fi
done

echo "[turflash] nvcc=$(command -v nvcc || true)"
echo "[turflash] ptxas=$(command -v ptxas || true)"
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
