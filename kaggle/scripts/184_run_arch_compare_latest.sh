#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${Y13_ROOT:-$HOME/kaggle/work_here/yolov13}"
PY="${REPO_ROOT}/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT="${Y13_ARCH_COMPARE_OUT:-$HOME/kaggle/working/arch_compare_latest}"
DATA="${Y13_ARCH_COMPARE_DATA:-$HOME/kaggle/work_here/dataset/converted_dataset/data.yaml}"
EPOCHS="${Y13_ARCH_COMPARE_EPOCHS:-5}"
IMGSZ="${Y13_ARCH_COMPARE_IMGSZ:-240}"
BATCH="32"
FRACTION="${Y13_ARCH_COMPARE_FRACTION:-0.50}"
WORKERS="${Y13_ARCH_COMPARE_WORKERS:-24}"
PREFETCH="${Y13_ARCH_COMPARE_PREFETCH:-8}"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python: ${PY}" >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "Dataset YAML not found: ${DATA}" >&2
  exit 1
fi

mkdir -p "${OUT}"
: > "${OUT}/run_durations.csv"
echo "[arch-compare] started $(date -Is)" | tee "${OUT}/run.log"

COMMON_ARGS=(
  --data "${DATA}"
  --epochs "${EPOCHS}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --device 0
  --workers "${WORKERS}"
  --project "${OUT}"
  --arg "fraction=${FRACTION}"
  --arg "prefetch_factor=${PREFETCH}"
  --arg "persistent_workers=true"
  --arg "classes=[0,9]"
  --arg "plots=true"
  --arg "exist_ok=true"
)

run_one() {
  local run_name="$1"
  local model_path="$2"
  local flash_mode="$3"
  local log_path="${OUT}/${run_name}.log"

  echo "[arch-compare] run=${run_name} model=${model_path} flash=${flash_mode}" | tee -a "${OUT}/run.log"
  local start_ts end_ts rc
  start_ts="$(date +%s)"

  "${PY}" "${REPO_ROOT}/scripts/train.py" \
    --model "${model_path}" \
    --name "${run_name}" \
    --flash-mode "${flash_mode}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${log_path}"

  rc="${PIPESTATUS[0]}"
  end_ts="$(date +%s)"
  echo "${run_name},${rc},$((end_ts - start_ts))" >> "${OUT}/run_durations.csv"

  if [[ "${rc}" -ne 0 ]]; then
    echo "[arch-compare] run failed: ${run_name}" | tee -a "${OUT}/run.log"
    return "${rc}"
  fi
}

run_one "baseline_flash_off" "ultralytics/cfg/models/v13/yolov13l.yaml" "fallback"
run_one "baseline_flash_on" "ultralytics/cfg/models/v13/yolov13l.yaml" "auto"
run_one "v2_flash_off" "ultralytics/cfg/models/v13/yolov13l_2.yaml" "fallback"
run_one "v2_flash_on" "ultralytics/cfg/models/v13/yolov13l_2.yaml" "auto"

"${PY}" "${REPO_ROOT}/kaggle/scripts/184_generate_arch_compare_plots.py" \
  --out-root "${OUT}" \
  2>&1 | tee "${OUT}/plot_generation.log"

echo "[arch-compare] finished $(date -Is)" | tee -a "${OUT}/run.log"
