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
2. NVIDIA driver check/install step (`27_install_nvidia_driver_535.sh`)
3. optional Roboflow dataset prep (`15_roboflow_ready.sh`) when `Y13_AUTO_ROBOFLOW_READY=1`
4. dependency install (`20_install_deps.sh`)
5. CUDA/GPU checks (`30_gpu_check.sh`, `32_cuda_sanity_report.sh`)
6. optional DDP smoke train (`40_ddp_smoke.sh`)
7. zip packaging (`50_package_zip.sh`)

Output zip:

- `/kaggle/working/yolov13.zip`

## 3) Manual setup (step-by-step)

If you prefer explicit setup:

```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/10_setup_uv.sh
bash kaggle/scripts/27_install_nvidia_driver_535.sh
bash kaggle/scripts/20_install_deps.sh
bash kaggle/scripts/30_gpu_check.sh
bash kaggle/scripts/32_cuda_sanity_report.sh
```

Torch stack defaults to latest validated runtime (`torch==2.11.0`, `torchvision==0.26.0`, `nvidia-nccl-cu13==2.29.7`).
Optional override before `20_install_deps.sh`:

```bash
export Y13_TORCH_VERSION=2.5.1
export Y13_TORCHVISION_VERSION=0.20.1
export Y13_NCCL_VERSION=2.29.7
export Y13_TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
```

For latest cu13 stack on hosts with older system CUDA toolkits, turFlash installer also injects NVCC from pip (`nvidia-cuda-nvcc`) and prepends it to `PATH`.
It also installs matching CUDA headers (`nvidia-cuda-cccl`) and builds turFlash with `--no-deps` to avoid torch/NCCL downgrades.

### 3.1) Optional one-shot Roboflow dataset prep

Use this helper when you want a fast repeatable download/extract into `/kaggle/work_here/datasets`.

```bash
cd /kaggle/work_here/yolov13

# default dataset key/url from project docs
bash kaggle/scripts/15_roboflow_ready.sh
```

Useful env flags:

- `Y13_ROBOFLOW_FORCE=1` re-download/re-extract
- `Y13_ROBOFLOW_DATASET_NAME=...` custom destination folder name
- `Y13_ROBOFLOW_REMAP_STUDENT_TEACHER=1` auto-run class remap (`student`, `teacher`)
- `Y13_AUTO_ROBOFLOW_READY=1` run this script automatically from `run_all.sh`

## 4) Flash backend controls (global)

Default behavior in this repo prefers Turing flash on T4 when available.

Naming note:

- In this repo/docs, `turFlash` refers to the Turing-focused package `flash-attention-turing`.
- Python import/module name is `flash_attn_turing`.
- Runtime backend string in logs is `flash_attn_turing`.
- See detailed research and gap analysis: `roadmap/phase3_upgrade/TURFLASH_T4_RESEARCH_AND_GAP_ANALYSIS.md`.

### 4.1 Install / build Turing FlashAttention

The repo supports installing `ssiu/flash-attention-turing` through:

- `kaggle/scripts/20_install_deps.sh` (auto-calls `25_install_turing_flash.sh` when flags are enabled)
- or direct call to `kaggle/scripts/25_install_turing_flash.sh`

Recommended setup sequence:

```bash
cd /kaggle/work_here/yolov13
source .venv/bin/activate

# enable install + runtime preference
export Y13_INSTALL_TURING_FLASH=1
export Y13_USE_TURING_FLASH=1
export Y13_DISABLE_FLASH=0

# install project deps and (if enabled) build flash-attention-turing
bash kaggle/scripts/20_install_deps.sh
```

Notes:

- installer worktree path: `/kaggle/work_here/flash-attention-turing`
- build is best-effort; if it fails, repo falls back to non-flash attention path
- you can rerun only Turing install with:

```bash
cd /kaggle/work_here/yolov13
export Y13_INSTALL_TURING_FLASH=1
bash kaggle/scripts/25_install_turing_flash.sh
```

### 4.2 Verify installation and backend resolution

```bash
cd /kaggle/work_here/yolov13
source .venv/bin/activate

# verify Python package import
python - <<'PY'
import importlib
mod = importlib.import_module('flash_attn_turing')
print('flash_attn_turing import ok:', mod is not None)
PY

# verify backend picked by repo attention selector
Y13_USE_TURING_FLASH=1 Y13_DISABLE_FLASH=0 python - <<'PY'
from ultralytics.nn.modules import block
block.configure_flash_backend(disable_flash=False, use_turing_flash=True)
print('FLASH_BACKEND =', getattr(block, 'FLASH_BACKEND', 'unknown'))
PY
```

Expected backend on T4 with successful build:

- `FLASH_BACKEND = flash_attn_turing`

### 4.3 Runtime controls during train/val/test/export/benchmark

```bash
# preferred Turing path
export Y13_USE_TURING_FLASH=1
export Y13_DISABLE_FLASH=0

# force fallback (debug / compare)
# export Y13_DISABLE_FLASH=1
```

For script-level control, pass `--flash-mode`:

- old style: `scripts/train.py`, `scripts/val.py`, `scripts/test.py`, `scripts/export.py`, `scripts/benchmark.py`
- new API-style: `scripts/api_style/*.py`

Modes:

- `--flash-mode turing`
- `--flash-mode auto`
- `--flash-mode fallback`

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
- runs feature-map projection only after training completes, using `best.pt`

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

### F) Optional: use post-train feature projection flags directly in `scripts/train.py`

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --task detect \
  --time 2 \
  --imgsz 640 \
  --batch 16 \
  --device 0,1 \
  --workers 8 \
  --cache ram \
  --fraction 1 \
  --optimizer MuSGD \
  --flash-mode turing \
  --feature-projection \
  --feature-projection-script kaggle/scripts/37_feature_map_projection.py \
  --feature-projection-valid-dir /kaggle/work_here/datasets/my_detect/valid/images \
  --feature-projection-out-dir /kaggle/working/custom_runs/detect_l_time2_custom/feature_projection \
  --feature-projection-md-path /kaggle/working/custom_runs/detect_l_time2_custom/ff_maps.md \
  --feature-projection-log-path /kaggle/working/custom_runs/detect_l_time2_custom/feature_projection.log \
  --feature-projection-flash-mode same \
  --project /kaggle/working/custom_runs \
  --name detect_l_time2_custom
```

`--feature-projection-flash-mode same` reuses training flash mode (e.g., Turing).

## 6.2) End-to-end custom dataset example (setup -> train -> val -> test -> benchmark)

This is a full developer flow you can repeat for any custom YOLO-format detect dataset.

### A) Setup from scratch

```bash
mkdir -p /kaggle/work_here
cd /kaggle/work_here

if [ ! -d yolov13 ]; then
  git clone https://github.com/ahmedelbamby-aast/yolov13-custom.git yolov13
fi

cd /kaggle/work_here/yolov13
bash kaggle/scripts/10_setup_uv.sh
bash kaggle/scripts/20_install_deps.sh
source .venv/bin/activate
```

### B) Optional: sync latest run scripts from `origin/main`

Use this when your local Kaggle clone is not on the latest main commit.

```bash
cd /kaggle/work_here/yolov13
git fetch origin
git show origin/main:kaggle/scripts/37_feature_map_projection.py > kaggle/scripts/37_feature_map_projection.py
git show origin/main:kaggle/scripts/run_custom_time2_tmp.sh > kaggle/scripts/run_custom_time2_tmp.sh
chmod +x kaggle/scripts/run_custom_time2_tmp.sh
```

### C) Prepare custom dataset

Expected structure:

- `/kaggle/work_here/datasets/my_detect/train/images`
- `/kaggle/work_here/datasets/my_detect/train/labels`
- `/kaggle/work_here/datasets/my_detect/valid/images`
- `/kaggle/work_here/datasets/my_detect/valid/labels`
- `/kaggle/work_here/datasets/my_detect/test/images`
- `/kaggle/work_here/datasets/my_detect/test/labels`
- `/kaggle/work_here/datasets/my_detect/data.yaml`

Example `data.yaml`:

```yaml
path: /kaggle/work_here/datasets/my_detect
train: train/images
val: valid/images
test: test/images
nc: 2
names: [student, teacher]
```

### D) Train (DDP 2xT4 + Turing flash)

Notes:

- `amp=True` is the recommended train-time mixed precision path (half-precision behavior for training).
- DDP is enabled by `--device 0,1`.

```bash
cd /kaggle/work_here/yolov13
source .venv/bin/activate

export Y13_DISABLE_FLASH=0
export Y13_USE_TURING_FLASH=1

python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
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
  --flash-mode turing \
  --amp true \
  --project /kaggle/working/custom_runs \
  --name detect_l_time2_custom \
  | tee /kaggle/working/custom_runs/detect_l_time2_custom_train.log
```

### E) Validate (`val` split)

```bash
python scripts/val.py \
  --model /kaggle/working/custom_runs/detect_l_time2_custom/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --flash-mode turing
```

### F) Test (`test` split)

```bash
python scripts/test.py \
  --model /kaggle/working/custom_runs/detect_l_time2_custom/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split test \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

### G) Benchmark (controlled ONNX + TensorRT on T4)

```bash
python scripts/benchmark.py \
  --model /kaggle/working/custom_runs/detect_l_time2_custom/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --imgsz 640 \
  --device 0 \
  --half \
  --flash-mode turing \
  --format onnx \
  --format engine \
  --out-json /kaggle/working/custom_runs/detect_l_time2_custom/bench_t4.json
```

### H) Quick verification checklist

```bash
grep -nE "flash_mode_env|resolved_flash_backend|DDP|time=2|optimizer=MuSGD|cache=ram|fraction=1" /kaggle/working/custom_runs/detect_l_time2_custom_train.log
ls -la /kaggle/working/custom_runs/detect_l_time2_custom/weights
cat /kaggle/working/custom_runs/detect_l_time2_custom/results.csv | tail -n 5
```

If you want the integrated final-run package (train + feature-map projection + `ff_maps.md`), use section **6.1** launcher:

```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/run_custom_time2_tmp.sh
```

### 6.2.1) Fix class-ID mismatch quickly (new helper script)

If you see warnings like:

- `Label class 9 exceeds dataset class count 2`

then your `data.yaml` was reduced (e.g., to 2 classes) but label `.txt` files still contain old IDs.

Use `kaggle/scripts/38_class_remap.py` to keep selected classes and remap them to `0..N-1`.

Keep classes by **name** (recommended):

```bash
cd /kaggle/work_here/yolov13
source .venv/bin/activate

python kaggle/scripts/38_class_remap.py \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --include-name student \
  --include-name teacher
```

Keep classes by **original ID**:

```bash
python kaggle/scripts/38_class_remap.py \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --include-id 8 \
  --include-id 9
```

Preview only (no write):

```bash
python kaggle/scripts/38_class_remap.py \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --include-name student \
  --include-name teacher \
  --dry-run
```

What the script does:

- rewrites all `labels/*.txt` across train/val/valid/test
- drops boxes from non-selected classes
- remaps selected classes to contiguous IDs starting at 0
- updates `data.yaml` `names` and `nc`
- deletes stale `*.cache` files by default (use `--keep-cache` to disable)

After remap, rerun val first (single GPU) before DDP:

```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/val.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split val \
  --device 0
```

## 6.3) Full pipeline example (old-style scripts)

This is a compact copy-paste workflow from setup to train/val/test/export/benchmark using `scripts/*.py`.

```bash
# 1) Setup
mkdir -p /kaggle/work_here
cd /kaggle/work_here
if [ ! -d yolov13 ]; then
  git clone https://github.com/ahmedelbamby-aast/yolov13-custom.git yolov13
fi
cd /kaggle/work_here/yolov13
bash kaggle/scripts/10_setup_uv.sh
bash kaggle/scripts/20_install_deps.sh
source .venv/bin/activate

# 2) Flash runtime
export Y13_DISABLE_FLASH=0
export Y13_USE_TURING_FLASH=1

# 3) Train (DDP 2xT4)
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
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
  --flash-mode turing \
  --amp true \
  --project /kaggle/working/custom_runs \
  --name old_style_detect

# 4) Validate
python scripts/val.py \
  --model /kaggle/working/custom_runs/old_style_detect/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --flash-mode turing

# 5) Test
python scripts/test.py \
  --model /kaggle/working/custom_runs/old_style_detect/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split test \
  --imgsz 640 \
  --batch 16 \
  --device 0

# 6) Export
python scripts/export.py \
  --model /kaggle/working/custom_runs/old_style_detect/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --dynamic \
  --flash-mode turing

# 7) Benchmark
python scripts/benchmark.py \
  --model /kaggle/working/custom_runs/old_style_detect/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --imgsz 640 \
  --device 0 \
  --half \
  --flash-mode turing \
  --format onnx \
  --format engine \
  --out-json /kaggle/working/custom_runs/old_style_detect/bench_t4.json
```

## 6.4) Full pipeline example (new API-style scripts)

This is the equivalent workflow using `scripts/api_style/*.py`.

```bash
# 1) Setup
mkdir -p /kaggle/work_here
cd /kaggle/work_here
if [ ! -d yolov13 ]; then
  git clone https://github.com/ahmedelbamby-aast/yolov13-custom.git yolov13
fi
cd /kaggle/work_here/yolov13
bash kaggle/scripts/10_setup_uv.sh
bash kaggle/scripts/20_install_deps.sh
source .venv/bin/activate

# 2) Flash runtime
export Y13_DISABLE_FLASH=0
export Y13_USE_TURING_FLASH=1

# 3) Train (DDP 2xT4, API-style)
python scripts/api_style/train_api.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
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
  --flash-mode turing \
  --project /kaggle/working/custom_runs \
  --name api_style_detect

# 4) Validate
python scripts/api_style/val_api.py \
  --model /kaggle/working/custom_runs/api_style_detect/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --flash-mode turing

# 5) Test
python scripts/api_style/test_api.py \
  --model /kaggle/working/custom_runs/api_style_detect/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split test \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --flash-mode turing

# 6) Export
python scripts/api_style/export_api.py \
  --model /kaggle/working/custom_runs/api_style_detect/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --dynamic \
  --flash-mode turing

# 7) Benchmark
python scripts/api_style/benchmark_api.py \
  --model /kaggle/working/custom_runs/api_style_detect/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --imgsz 640 \
  --device 0 \
  --format onnx \
  --flash-mode turing \
  --out-json /kaggle/working/custom_runs/api_style_detect/bench_t4_api_style.json
```

If your dataset has no `test` split, run `--split val` with `scripts/api_style/test_api.py`.

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

### Dirty-data head_dim=32 smoke benchmark (custom dataset)

Runs baseline vs head32-enabled on a remapped dirty dataset subset, then writes JSON + report + plot.

```bash
cd /kaggle/work_here/yolov13
source .venv/bin/activate

# optional: prepare + normalize + remap Roboflow dataset first
Y13_ROBOFLOW_FORCE=1 \
Y13_ROBOFLOW_REMAP_STUDENT_TEACHER=1 \
bash kaggle/scripts/15_roboflow_ready.sh

# benchmark 5 epochs using 5% subset
Y13_DIRTY_DATA_YAML=/kaggle/work_here/datasets/roboflow_custom_detect_dirty/data.yaml \
Y13_HEAD32_BENCH_EPOCHS=5 \
Y13_HEAD32_BENCH_FRACTION=0.05 \
Y13_HEAD32_BENCH_BASE_DEVICE=1 \
Y13_HEAD32_BENCH_HEAD32_DEVICE=0 \
Y13_HEAD32_BENCH_WORKERS_PER_RUN=2 \
bash kaggle/scripts/182_benchmark_head32_dirty_smoke.sh
```

Benchmark policy used in this phase:

- real dirty dataset only (no synthetic benchmark dataset)
- two runs in parallel:
  - baseline on `gpu:1`
  - head32-enabled on `gpu:0`
- each run is hard-pinned via `CUDA_VISIBLE_DEVICES` and launched with local `--device 0`
- workers per run: `2` (total `4`)
- train cache: `ram`

Artifacts:

- `kaggle/benchmarks/flash_head32_dirty_smoke/compare_summary.json`
- `kaggle/benchmarks/flash_head32_dirty_smoke/REPORT.md`
- `kaggle/benchmarks/flash_head32_dirty_smoke/telemetry_compare.png`

Round-separated benchmark archives and comparison plots:

- `kaggle/benchmarks/flash_head32_dirty_smoke_round1/`
- `kaggle/benchmarks/flash_head32_dirty_smoke_round2/`
- `kaggle/benchmarks/flash_head32_dirty_smoke_comparison/`

Generate/update these round-separated plots and reports:

```bash
python kaggle/scripts/183_generate_dirty_rounds_plots.py
```

Latest 5-epoch dirty-data smoke snapshot (fraction `0.05`):

- parallel policy: baseline `gpu:1`, head32 `gpu:0`, workers per run `2`, cache `ram`
- baseline train epoch time: `133.20s`
- head32-enabled train epoch time: `133.20s`
- train speedup: `1.00x`
- baseline CUDA-only hit-rate: `0.00%` (`0/10128`)
- head32 CUDA-only hit-rate: `100.00%` (`10128/10128`)
- baseline fallback reasons: `{'not_cuda': 24, 'unsupported_head_dim_32': 10128}`
- head32-enabled fallback reasons: `{'not_cuda': 24}`

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
- turFlash/T4 research and gap analysis: `roadmap/phase3_upgrade/TURFLASH_T4_RESEARCH_AND_GAP_ANALYSIS.md`
