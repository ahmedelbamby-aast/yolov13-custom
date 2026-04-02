#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/00_entry_banner.py"

bash "${SCRIPT_DIR}/10_setup_uv.sh"
bash "${SCRIPT_DIR}/20_install_deps.sh"
bash "${SCRIPT_DIR}/30_gpu_check.sh"

if [[ "${RUN_DDP_SMOKE:-1}" == "1" ]]; then
  bash "${SCRIPT_DIR}/40_ddp_smoke.sh"
fi

if [[ "${RUN_FULL_VALIDATION:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/100_full_validation.sh"
fi

bash "${SCRIPT_DIR}/50_package_zip.sh"

echo "Pipeline complete: /kaggle/working/yolov13.zip"
