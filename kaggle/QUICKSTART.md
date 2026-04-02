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

## Optional Flash Backend Flags (T4/Turing)
```bash
# default runtime uses Turing flash on T4
export Y13_INSTALL_TURING_FLASH=1   # one-time install/build
# export Y13_USE_TURING_FLASH=0     # disable if needed
# export Y13_DISABLE_FLASH=1        # force fallback backend
```

## Kaggle Notebook UI (No SSH)

Use prebuilt notebooks in `notebooks/`:
- `notebooks/01_train.ipynb`
- `notebooks/02_validate.ipynb`
- `notebooks/03_export.ipynb`
- `notebooks/04_tracking.ipynb`
- `notebooks/05_test.ipynb`

All notebooks use one canonical environment in `/kaggle/work_here/yolov13/.venv`.

## Packaging only
```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/50_package_zip.sh
```
