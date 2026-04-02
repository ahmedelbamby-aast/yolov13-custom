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
export MAX_JOBS=2

uv pip install --python "${PY}" ninja setuptools wheel

set +e
uv pip install --python "${PY}" -v --no-build-isolation "${WORKTREE}"
rc=$?
set -e

if [[ "${rc}" -ne 0 ]]; then
  echo "[warn] Turing FlashAttention build failed; fallback backend remains active."
else
  echo "Turing FlashAttention installed successfully."
fi
