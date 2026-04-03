#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/work_here/yolov13"
PY="${REPO_ROOT}/.venv/bin/python"
DATA_YAML="/kaggle/work_here/datasets/roboflow_custom_detect/data.yaml"
VALID_DIR="/kaggle/work_here/datasets/roboflow_custom_detect/valid/images"
OUT_ROOT="/kaggle/working/final_run"
RUN_NAME="detect_l_time2_musgd_ddp"
TRAIN_LOG="${OUT_ROOT}/train.log"
FMAP_LOG="${OUT_ROOT}/feature_projection.log"
BEST_PT="${OUT_ROOT}/${RUN_NAME}/weights/best.pt"

mkdir -p "${OUT_ROOT}"

pkill -f "scripts/train.py" || true
pkill -f "37_feature_map_projection.py" || true

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="$(nproc)"
export MKL_NUM_THREADS="$(nproc)"
export Y13_DISABLE_FLASH=0
export Y13_USE_TURING_FLASH=1

"${PY}" scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data "${DATA_YAML}" \
  --task detect \
  --time 2 \
  --epochs 1000 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --cache ram \
  --fraction 1 \
  --device 0,1 \
  --optimizer MuSGD \
  --project "${OUT_ROOT}" \
  --name "${RUN_NAME}" \
  --flash-mode turing \
  --feature-projection \
  --feature-projection-script kaggle/scripts/37_feature_map_projection.py \
  --feature-projection-valid-dir "${VALID_DIR}" \
  --feature-projection-device 0 \
  --feature-projection-out-dir "${OUT_ROOT}/feature_projection" \
  --feature-projection-md-path "${OUT_ROOT}/ff_maps.md" \
  --feature-projection-log-path "${FMAP_LOG}" \
  --feature-projection-dataset-name "SBAS-AASTMT-AI-ALAMEIN" \
  --feature-projection-model-name "YOLOv13" \
  --feature-projection-variant "l" \
  --feature-projection-flash-mode same \
  --arg val=true \
  --arg plots=true \
  --arg verbose=true \
  > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!

sleep 15
grep -nE "flash_mode_env|resolved_flash_backend|DDP|world_size" "${TRAIN_LOG}" || true

grep -n "enlargement_enabled" kaggle/scripts/37_feature_map_projection.py
grep -n "cv2.resize" kaggle/scripts/37_feature_map_projection.py

printf 'training_pid=%s\nfeature_projection_pid=%s\nout_root=%s\ntrain_log=%s\nfeature_projection_log=%s\n' \
  "${TRAIN_PID}" "embedded_post_train" "${OUT_ROOT}" "${TRAIN_LOG}" "${FMAP_LOG}"
