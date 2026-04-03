# YOLOv13 Quickstart (Kaggle + Dev Workflows)

This guide covers setup and end-to-end usage for:

- environment preparation
- training
- validation/testing
- prediction
- export
- benchmarking
- packaging

It includes both:

- Kaggle automation scripts (`kaggle/scripts/*`)
- modular developer scripts (`scripts/*`)

## 1) Paths and assumptions

- Repo workspace: `/kaggle/work_here/yolov13`
- Output root: `/kaggle/working`
- Python venv: `/kaggle/work_here/yolov13/.venv`

If you are on a fresh session:

```bash
mkdir -p /kaggle/work_here
cd /kaggle/work_here
git clone https://github.com/ahmedelbamby-aast/yolov13-custom.git yolov13
cd /kaggle/work_here/yolov13
```

## 2) One-command bootstrap pipeline

Run the full bootstrap pipeline:

```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/run_all.sh
```

This performs:

1. venv setup (`10_setup_uv.sh`)
2. dependency install (`20_install_deps.sh`)
3. NVIDIA driver check/install step (`27_install_nvidia_driver_535.sh`)
4. CUDA/GPU checks (`30_gpu_check.sh`, `32_cuda_sanity_report.sh`)
5. optional DDP smoke train (`40_ddp_smoke.sh`)
6. zip packaging (`50_package_zip.sh`)

Output zip:

- `/kaggle/working/yolov13.zip`

## 3) Manual setup (step-by-step)

If you prefer explicit setup:

```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/10_setup_uv.sh
bash kaggle/scripts/20_install_deps.sh
bash kaggle/scripts/27_install_nvidia_driver_535.sh
bash kaggle/scripts/30_gpu_check.sh
bash kaggle/scripts/32_cuda_sanity_report.sh
```

## 4) Flash backend controls (global)

Default behavior in this repo prefers Turing flash on T4 when available.

```bash
# optional one-time install/build of turing flash
export Y13_INSTALL_TURING_FLASH=1

# flash runtime control
export Y13_USE_TURING_FLASH=1   # enable turing path
export Y13_DISABLE_FLASH=0      # do not force fallback

# force fallback backend
# export Y13_DISABLE_FLASH=1
```

## 5) Core developer workflows (recommended)

Use the modular scripts under `scripts/`.

### 5.1 Training

```bash
cd /kaggle/work_here/yolov13
source .venv/bin/activate

python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data coco8.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0,1 \
  --workers 4 \
  --project /kaggle/working/runs/train \
  --name detect_l_50e \
  --flash-mode turing
```

### 5.2 Validation

```bash
python scripts/val.py \
  --model /kaggle/working/runs/train/detect_l_50e/weights/best.pt \
  --data coco8.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --flash-mode auto
```

### 5.3 Testing (val alias)

```bash
python scripts/test.py \
  --model /kaggle/working/runs/train/detect_l_50e/weights/best.pt \
  --data coco8.yaml \
  --split test \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

### 5.4 Prediction

```bash
python scripts/predict.py \
  --model /kaggle/working/runs/train/detect_l_50e/weights/best.pt \
  --source /kaggle/work_here/datasets/coco8/images/val \
  --imgsz 640 \
  --conf 0.25 \
  --device 0 \
  --save \
  --project /kaggle/working/runs/predict \
  --name detect_l_pred \
  --flash-mode fallback
```

### 5.5 Export

```bash
python scripts/export.py \
  --model /kaggle/working/runs/train/detect_l_50e/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --dynamic
```

### 5.6 Benchmark

```bash
python scripts/benchmark.py \
  --model /kaggle/working/runs/train/detect_l_50e/weights/best.pt \
  --data coco8.yaml \
  --imgsz 640 \
  --device 0 \
  --half \
  --flash-mode turing \
  --format onnx \
  --format engine \
  --out-json /kaggle/working/bench_detect_t4.json
```

Use GPU (`--device 0`) for benchmarks. The script runs controlled formats to keep results focused on T4 CUDA paths.
For stability, ONNX runs in FP32 while TensorRT/TorchScript can use FP16.

## 6) Custom dataset usage

All scripts accept custom dataset YAML through `--data`:

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l-seg.yaml \
  --data /kaggle/work_here/mydata/custom_seg.yaml \
  --task segment \
  --epochs 100 \
  --imgsz 640 \
  --device 0,1
```

You can pass any extra Ultralytics override via repeated `--arg KEY=VALUE`.

You can also pass Ultralytics mode args directly in CLI style (`--key value` or `--key=value`).

Reference: https://docs.ultralytics.com/modes/

Example:

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l-pose.yaml \
  --data /kaggle/work_here/mydata/custom_pose.yaml \
  --task pose \
  --flash-mode turing \
  --arg optimizer=AdamW \
  --arg lr0=0.001 \
  --arg patience=50
```

Direct mode-arg example:

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data /kaggle/work_here/mydata/custom_det.yaml \
  --epochs 300 \
  --optimizer AdamW \
  --lr0 0.001 \
  --weight_decay 0.0005 \
  --close_mosaic 10 \
  --cos_lr true
```

## 6.1) Repeatable final detect run (2xT4 DDP + Turing flash + feature-map projection)

Use this procedure when you want to re-run the exact final workflow:

- task: detect only
- model: YOLOv13-l
- optimizer: MuSGD
- time budget: 2 hours (`time=2`)
- DDP: `device=0,1` (2xT4)
- data loading: `cache=ram`, `workers=8`, `fraction=1`
- outputs: `/kaggle/working/final_run`
- feature-map overlays + markdown: `/kaggle/working/final_run/feature_projection` and `/kaggle/working/final_run/ff_maps.md`

### A) Sync latest run scripts to Kaggle repo

If your Kaggle workspace is not on `main`, pull the two scripts directly from `origin/main`:

```bash
cd /kaggle/work_here/yolov13
git fetch origin
git show origin/main:kaggle/scripts/37_feature_map_projection.py > kaggle/scripts/37_feature_map_projection.py
git show origin/main:kaggle/scripts/run_custom_time2_tmp.sh > kaggle/scripts/run_custom_time2_tmp.sh
chmod +x kaggle/scripts/run_custom_time2_tmp.sh
```

### B) Enforce dataset classes for this run

Roboflow YAML must be reduced to only `student` and `teacher`:

```bash
/kaggle/work_here/yolov13/.venv/bin/python - <<'PY'
import pathlib
import yaml

p = pathlib.Path('/kaggle/work_here/datasets/roboflow_custom_detect/data.yaml')
d = yaml.safe_load(p.read_text())
d['nc'] = 2
d['names'] = ['student', 'teacher']
p.write_text(yaml.safe_dump(d, sort_keys=False))
print('dataset_yaml', p)
print('nc', d['nc'])
print('names', d['names'])
PY
```

### C) Launch the run

The launcher does all of this automatically:

- kills any previous `scripts/train.py` and `37_feature_map_projection.py`
- sets `Y13_DISABLE_FLASH=0` and `Y13_USE_TURING_FLASH=1`
- starts train with `--flash-mode turing`
- starts feature-map projection in parallel (waits for `best.pt`)

```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/run_custom_time2_tmp.sh
```

### D) Verify critical runtime conditions

Check Turing backend, DDP, and run arguments:

```bash
grep -nE "flash_mode_env|resolved_flash_backend|DDP|time=2|batch=16|imgsz=640|optimizer=MuSGD|cache=ram|fraction=1" /kaggle/working/final_run/train.log
```

Expected key lines include:

- `Y13_USE_TURING_FLASH=1`
- `resolved_flash_backend=flash_attn_turing`
- DDP launch with `--nproc_per_node 2`

### E) Monitor and collect outputs

```bash
ps -ef | grep -E "scripts/train.py|37_feature_map_projection.py|torch.distributed.run" | grep -v grep
tail -n 50 /kaggle/working/final_run/train.log
tail -n 50 /kaggle/working/final_run/feature_projection.log
ls -la /kaggle/working/final_run
```

At completion, verify:

- weights/checkpoints under `/kaggle/working/final_run/detect_l_time2_musgd_ddp/`
- per-layer projected overlays under `/kaggle/working/final_run/feature_projection/`
- final markdown report `/kaggle/working/final_run/ff_maps.md`

## 7) Kaggle validation and utility pipelines

### DDP smoke

```bash
bash kaggle/scripts/40_ddp_smoke.sh
```

### 5-epoch DDP train

```bash
bash kaggle/scripts/60_ddp_train_5epochs.sh
```

### Full validation pipeline

```bash
bash kaggle/scripts/100_full_validation.sh
```

### Export validation helper

```bash
bash kaggle/scripts/70_export_onnx_tensorrt.sh
```

## 8) L-scale fallback vs Turing benchmark suite

Run the dedicated multi-task comparison benchmark:

```bash
cd /kaggle/work_here/yolov13
Y13_BENCH_EPOCHS=30 \
Y13_BENCH_WORKERS=4 \
Y13_BENCH_OUT_ROOT=/kaggle/working/phase2_l_flash_compare_v2 \
bash kaggle/scripts/181_benchmark_l_flash_tasks.sh
```

Main outputs:

- `/kaggle/working/phase2_l_flash_compare_v2/fallback/suite_summary.json`
- `/kaggle/working/phase2_l_flash_compare_v2/turing/suite_summary.json`
- `/kaggle/working/phase2_l_flash_compare_v2/compare_summary.json`
- `/kaggle/working/phase2_l_flash_compare_v2/plots/`

Synced report/artifacts in repo:

- `kaggle/reports/BENCHMARK_L_FLASH_TASKS_COMPARISON.md`
- `kaggle/benchmarks/l_flash_tasks/`

## 9) Packaging and snapshots

### Create repo zip

```bash
bash kaggle/scripts/50_package_zip.sh
```

Output:

- `/kaggle/working/yolov13.zip`

### Optional full snapshot + custom bundles

```bash
python - <<'PY'
import shutil
from pathlib import Path
src = Path('/kaggle/work_here/yolov13')
dst = Path('/kaggle/working/yolov13_snapshot_full')
shutil.rmtree(dst, ignore_errors=True)
ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '.git', '.pytest_cache', 'ultralytics.egg-info')
shutil.copytree(src, dst, ignore=ignore)
print(dst)
PY

cd /kaggle/working
zip -r yolov13_snapshot_full.zip yolov13_snapshot_full
zip -r yolov13_l_flash_compare_bundle.zip \
  phase2_l_flash_compare_v2 \
  yolov13_snapshot_full/kaggle/benchmarks/l_flash_tasks \
  yolov13_snapshot_full/kaggle/reports/BENCHMARK_L_FLASH_TASKS_COMPARISON.md
```

## 10) Troubleshooting

### CUDA unavailable in Python but `/dev/nvidia*` exists

```bash
apt-get update -y && apt-get install -y pciutils
ls -l /dev/nvidia*
nvidia-smi
lspci | grep -i nvidia || true
cat /proc/driver/nvidia/version || true

python - <<'PY'
import os
print('CUDA_VISIBLE_DEVICES =', os.getenv('CUDA_VISIBLE_DEVICES'))
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda avail:', torch.cuda.is_available())
    print('count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print('torch error:', e)
PY
```

### `gh` auth for PR/issue operations

```bash
gh auth login
gh auth status
```

## 11) Notebook workflow (no SSH)

Use notebooks from `notebooks/` with the same venv in `/kaggle/work_here/yolov13/.venv`:

- `notebooks/01_train.ipynb`
- `notebooks/02_validate.ipynb`
- `notebooks/03_export.ipynb`
- `notebooks/04_tracking.ipynb`
- `notebooks/05_test.ipynb`
- `notebooks/06_benchmark_flash.ipynb`

## 12) References

- Developer script guide: `scripts/README.md`
- L-task benchmark report: `kaggle/reports/BENCHMARK_L_FLASH_TASKS_COMPARISON.md`
