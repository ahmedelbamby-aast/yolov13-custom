#!/usr/bin/env bash
set -euo pipefail

REPO=/kaggle/work_here/yolov13
LOG=${1:-/kaggle/working/phase3_upgrade_upstream_import_$(date +%Y%m%d_%H%M%S).log}
ln -sfn "$LOG" /kaggle/working/logs_latest.txt

cd "$REPO"

{
  echo "[phase6] upstream import log=$LOG"
  echo "[phase6] start $(date -Is)"
  git fetch https://github.com/ultralytics/ultralytics.git --tags
  git cherry-pick 36560f2a3 0bd8a0970 fa3dd8d96 a48139972 7eaea3c62 3f6584ad3 bed8778ae
  echo "[phase6] cherry-pick complete"
  git status --short --branch
  echo "[phase6] end $(date -Is)"
} | tee -a "$LOG"
