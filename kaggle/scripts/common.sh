#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export Y13_ROOT="${Y13_ROOT:-${REPO_ROOT_DEFAULT}}"
export Y13_WORKDIR="${Y13_WORKDIR:-/kaggle/work_here}"
export Y13_OUTPUT_DIR="${Y13_OUTPUT_DIR:-/kaggle/working}"

# Flash backend controls
export Y13_USE_TURING_FLASH="${Y13_USE_TURING_FLASH:-0}"
export Y13_INSTALL_TURING_FLASH="${Y13_INSTALL_TURING_FLASH:-0}"
export Y13_DISABLE_FLASH="${Y13_DISABLE_FLASH:-0}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${Y13_ROOT}:${PYTHONPATH:-}"

mkdir -p "${Y13_WORKDIR}" "${Y13_OUTPUT_DIR}"
cd "${Y13_ROOT}"
