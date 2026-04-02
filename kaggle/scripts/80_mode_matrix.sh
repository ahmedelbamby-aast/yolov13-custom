#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
OUT="/kaggle/working/y13_runs/mode_matrix"
mkdir -p "${OUT}"

"${PY}" - <<'PY'
from ultralytics import YOLO

run_root = '/kaggle/working/y13_runs/mode_matrix'
weights = '/kaggle/working/y13_runs/ddp_5epochs/weights/best.pt'

model = YOLO(weights)

# val mode argument coverage
model.val(data='coco8.yaml', imgsz=320, batch=16, device=0, split='val', conf=0.001, iou=0.7, plots=True, project=run_root, name='val_mode', exist_ok=True)

# predict mode argument coverage
model.predict(source='/kaggle/work_here/datasets/coco8/images/val', imgsz=320, conf=0.25, iou=0.7, device=0, save=True, save_txt=True, save_conf=True, project=run_root, name='predict_mode', exist_ok=True)

# benchmark a second export setting for ONNX
model.export(format='onnx', imgsz=320, dynamic=False, simplify=False, opset=12)

print('Mode matrix completed')
PY
