#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

REPO="${Y13_ROOT:-/kaggle/work_here/yolov13}"
LOG=${1:-/kaggle/working/phase3_upgrade_gate_$(date +%Y%m%d_%H%M%S).log}
HEARTBEAT_INTERVAL="${Y13_PROGRESS_INTERVAL_S:-300}"
HEARTBEAT_JSONL="/kaggle/working/phase3_upgrade_gate_heartbeat.jsonl"
ln -sfn "$LOG" /kaggle/working/logs_latest.txt

cd "$REPO"
source .venv/bin/activate
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[phase6] virtual environment activation failed" >&2
  exit 1
fi

heartbeat_loop() {
  while true; do
    local ts
    ts="$(date -Is)"
    echo "[phase6][heartbeat] ${ts} running"
    printf '{"job":"phase6_gate_runner","status":"running","ts":"%s"}\n' "${ts}" >> "${HEARTBEAT_JSONL}"
    sleep "${HEARTBEAT_INTERVAL}"
  done
}

heartbeat_loop >> "$LOG" 2>&1 &
HB_PID=$!

cleanup() {
  local rc=$?
  if kill -0 "${HB_PID}" >/dev/null 2>&1; then
    kill "${HB_PID}" >/dev/null 2>&1 || true
  fi
  local ts
  ts="$(date -Is)"
  local status="completed"
  if [[ $rc -ne 0 ]]; then
    status="failed"
  fi
  printf '{"job":"phase6_gate_runner","status":"%s","ts":"%s","code":%d}\n' "${status}" "${ts}" "$rc" >> "${HEARTBEAT_JSONL}"
  return $rc
}
trap cleanup EXIT

{
  echo "[phase6] log=$LOG"
  echo "[phase6] heartbeat_interval_s=${HEARTBEAT_INTERVAL}"
  echo "[phase6] heartbeat_jsonl=${HEARTBEAT_JSONL}"
  echo "[phase6] start $(date -Is)"
  python3 kaggle/scripts/phase3_upgrade/00_alignment_schema_check.py
  python3 kaggle/scripts/phase3_upgrade/01_env_report.py
  python3 kaggle/scripts/phase3_upgrade/02_cli_python_parity_gate.py
  python3 kaggle/scripts/phase3_upgrade/03_stress_gate.py
  python3 kaggle/scripts/phase3_upgrade/04_autobatch_ddp_notes.py
  python3 kaggle/scripts/34_phase3_custom_delta_audit.py
  python3 kaggle/scripts/36_phase3_final_gate.py
  echo "[phase6] complete $(date -Is)"
} | tee -a "$LOG"
