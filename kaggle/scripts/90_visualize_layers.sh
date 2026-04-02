#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"

"${PY}" - <<'PY'
from pathlib import Path
from ultralytics import YOLO

weights = '/kaggle/working/y13_runs/ddp_5epochs/weights/best.pt'
out_dir = Path('/kaggle/working/y13_runs/layer_visuals')
out_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(weights)
# visualize=True saves intermediate feature visualizations
model.predict(
    source='/kaggle/work_here/datasets/coco8/images/val',
    imgsz=320,
    device=0,
    save=True,
    visualize=True,
    project=str(out_dir),
    name='predict_visualize',
    exist_ok=True,
)

print('Layer visualization completed:', out_dir)
PY
