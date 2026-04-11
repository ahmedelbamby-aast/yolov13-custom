#!/usr/bin/env bash
set -euo pipefail

REPO=/kaggle/work_here/yolov13
LOG=${1:-/kaggle/working/phase3_upgrade_gate_$(date +%Y%m%d_%H%M%S).log}
ln -sfn "$LOG" /kaggle/working/logs_latest.txt

cd "$REPO"
source .venv/bin/activate

{
  echo "[phase6] log=$LOG"
  echo "[phase6] start $(date -Is)"
  python3 kaggle/scripts/phase3_upgrade/01_env_report.py
  python3 kaggle/scripts/phase3_upgrade/02_cli_python_parity_gate.py
  python3 kaggle/scripts/phase3_upgrade/03_stress_gate.py
  python3 kaggle/scripts/phase3_upgrade/04_autobatch_ddp_notes.py
  echo "[phase6] complete $(date -Is)"
} | tee -a "$LOG"
