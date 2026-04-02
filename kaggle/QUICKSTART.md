# YOLOv13 Kaggle Quickstart (2x T4 DDP)

## Paths
- Development workspace: `/kaggle/work_here/yolov13`
- Outputs and artifacts: `/kaggle/working`
- Remote sync target: `github.com/ahmedelbamby-aast/yolov13-custom`

## End-to-end pipeline
```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/run_all.sh
```

This pipeline will:
1. Show a colorful Rich startup banner for Eng.Ahmed ElBamby.
2. Create a uv-based virtual environment.
3. Install core dependencies with edge-case handling.
4. Validate 2-GPU CUDA visibility.
5. Run a DDP smoke training job.
6. Package the project as `/kaggle/working/yolov13.zip`.

## Manual training example (DDP)
```bash
cd /kaggle/work_here/yolov13
source kaggle/scripts/common.sh
source .venv/bin/activate
python - <<'PY'
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v13/yolov13.yaml')
model.train(
    data='coco.yaml',
    epochs=300,
    imgsz=640,
    batch=64,
    device='0,1',
    workers=8,
    project='/kaggle/working/y13_runs',
    name='train_ddp',
    exist_ok=True,
)
PY
```

## Packaging only
```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/50_package_zip.sh
```

## Kaggle Notebook UI (No SSH)

Use prebuilt notebooks in `notebooks/`:

- `notebooks/01_train.ipynb`
- `notebooks/02_validate.ipynb`
- `notebooks/03_export.ipynb`
- `notebooks/04_tracking.ipynb`
- `notebooks/05_test.ipynb`

These notebooks assume repository source starts in `/kaggle/working`, then mirror active development to `/kaggle/work_here`, and keep outputs in `/kaggle/working`.
