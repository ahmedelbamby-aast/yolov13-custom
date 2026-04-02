#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv. Run 10_setup_uv.sh first." >&2
  exit 1
fi

if [[ "${Y13_BENCH_ENSURE_TURING_INSTALL:-1}" == "1" ]]; then
  Y13_INSTALL_TURING_FLASH=1 bash "${SCRIPT_DIR}/25_install_turing_flash.sh"
fi

export Y13_REPO_ROOT="${Y13_ROOT}"
export Y13_BENCH_OUT_ROOT="${Y13_BENCH_OUT_ROOT:-${Y13_OUTPUT_DIR}/phase2_l_flash_compare}"
export Y13_BENCH_EPOCHS="${Y13_BENCH_EPOCHS:-5}"
export Y13_BENCH_IMGSZ="${Y13_BENCH_IMGSZ:-640}"
export Y13_BENCH_WORKERS="${Y13_BENCH_WORKERS:-4}"

"${PY}" "${SCRIPT_DIR}/180_benchmark_l_flash_tasks.py"

echo "L-scale flash backend comparison complete: ${Y13_BENCH_OUT_ROOT}"
