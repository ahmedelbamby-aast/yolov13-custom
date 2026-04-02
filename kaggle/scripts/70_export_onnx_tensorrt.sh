#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
WEIGHTS="/kaggle/working/y13_runs/ddp_5epochs/weights/best.pt"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Weights not found: ${WEIGHTS}" >&2
  exit 1
fi

"${PY}" - <<'PY'
from ultralytics import YOLO

weights = '/kaggle/working/y13_runs/ddp_5epochs/weights/best.pt'
model = YOLO(weights)

print('Export ONNX...')
onnx_path = model.export(format='onnx', imgsz=320, dynamic=True, simplify=True, opset=12)
print('ONNX:', onnx_path)

print('Export TensorRT...')
trt_path = model.export(format='engine', imgsz=320, half=True, workspace=4)
print('TensorRT:', trt_path)
PY
