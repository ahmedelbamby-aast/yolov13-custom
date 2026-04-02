#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"

"${PY}" - <<'PY'
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v13/yolov13.yaml')
model.train(
    data='coco8.yaml',
    epochs=1,
    imgsz=64,
    batch=8,
    workers=2,
    device='0,1',
    project='/kaggle/working/y13_runs',
    name='ddp_smoke',
    exist_ok=True,
)
print('DDP smoke training completed')
PY
