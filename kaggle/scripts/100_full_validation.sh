#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

python "${SCRIPT_DIR}/95_issue_audit_report.py"
bash "${SCRIPT_DIR}/60_ddp_train_5epochs.sh"
bash "${SCRIPT_DIR}/70_export_onnx_tensorrt.sh"
bash "${SCRIPT_DIR}/80_mode_matrix.sh"
bash "${SCRIPT_DIR}/90_visualize_layers.sh"
bash "${SCRIPT_DIR}/50_package_zip.sh"

echo "Full validation pipeline completed."
