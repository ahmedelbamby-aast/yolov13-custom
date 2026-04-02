#!/usr/bin/env bash
set -euo pipefail

export Y13_ROOT="${Y13_ROOT:-/kaggle/work_here/yolov13}"
export Y13_WORKDIR="${Y13_WORKDIR:-/kaggle/work_here}"
export Y13_OUTPUT_DIR="${Y13_OUTPUT_DIR:-/kaggle/working}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${Y13_ROOT}:${PYTHONPATH:-}"

mkdir -p "${Y13_WORKDIR}" "${Y13_OUTPUT_DIR}"
cd "${Y13_ROOT}"
