#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/00_entry_banner.py"

bash "${SCRIPT_DIR}/10_setup_uv.sh"
bash "${SCRIPT_DIR}/27_install_nvidia_driver_535.sh"

if [[ "${Y13_AUTO_ROBOFLOW_READY:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/15_roboflow_ready.sh"
fi

bash "${SCRIPT_DIR}/20_install_deps.sh"
bash "${SCRIPT_DIR}/30_gpu_check.sh"
bash "${SCRIPT_DIR}/32_cuda_sanity_report.sh"

if [[ "${RUN_DDP_SMOKE:-1}" == "1" ]]; then
  bash "${SCRIPT_DIR}/40_ddp_smoke.sh"
fi

if [[ "${RUN_FULL_VALIDATION:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/100_full_validation.sh"
fi

bash "${SCRIPT_DIR}/50_package_zip.sh"

echo "Pipeline complete: /kaggle/working/yolov13.zip"
