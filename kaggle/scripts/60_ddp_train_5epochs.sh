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
    epochs=5,
    imgsz=320,
    batch=16,
    workers=4,
    device='0,1',
    optimizer='AdamW',
    lr0=1e-3,
    weight_decay=5e-4,
    project='/kaggle/working/y13_runs',
    name='ddp_5epochs',
    exist_ok=True,
    amp=True,
)
print('DDP 5-epoch training completed')
PY
