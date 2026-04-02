# Developer Scripts (Train/Test/Export/Benchmark/Validation)

This folder provides modular, direct scripts for common workflows.

The goal is to feel like normal Ultralytics usage, while making runtime behavior explicit and easy to toggle.

## Design

- Each script is standalone from a developer perspective.
- No script chaining is required.
- Common behavior (flash toggles, extra arg parsing, runtime banner) is centralized in `scripts/_common.py`.
- You can pass any custom dataset YAML (`--data path/to/data.yaml`).

## Scripts

- `scripts/train.py`: train models.
- `scripts/val.py`: validate models.
- `scripts/test.py`: alias to `val.py` for test-style usage.
- `scripts/predict.py`: run inference.
- `scripts/export.py`: export to ONNX/TensorRT/TorchScript/etc.
- `scripts/benchmark.py`: benchmark model/export performance.

## Flash Backend Control (Turing / Fallback / Auto)

All scripts accept:

- `--flash-mode auto`: default behavior.
- `--flash-mode fallback`: force fallback attention (`Y13_DISABLE_FLASH=1`).
- `--flash-mode turing`: force Turing flash path (`Y13_USE_TURING_FLASH=1`).

The scripts set env flags before importing `ultralytics`, then re-run backend selection so the choice is deterministic.

## Generic Extra Overrides

You can pass any additional Ultralytics argument with repeated `--arg KEY=VALUE`.

Examples:

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l.yaml \
  --data coco8.yaml \
  --epochs 50 \
  --flash-mode turing \
  --arg optimizer=AdamW \
  --arg lr0=0.001 \
  --arg patience=50
```

Values are parsed as Python literals where possible:

- `true/false` -> booleans
- numbers -> numeric
- lists/dicts like `"[1,2]"`, `"{'a':1}"`
- otherwise string

## Usage Examples

### 1) Train (custom data)

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/v13/yolov13l-seg.yaml \
  --data path/to/custom-seg.yaml \
  --task segment \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0,1 \
  --flash-mode turing
```

### 2) Validate

```bash
python scripts/val.py \
  --model runs/train/exp/weights/best.pt \
  --data path/to/custom.yaml \
  --split val \
  --imgsz 640 \
  --flash-mode auto
```

### 3) Test alias

```bash
python scripts/test.py \
  --model runs/train/exp/weights/best.pt \
  --data path/to/custom.yaml \
  --split test
```

### 4) Predict

```bash
python scripts/predict.py \
  --model runs/train/exp/weights/best.pt \
  --source path/to/images \
  --save \
  --flash-mode fallback
```

### 5) Export

```bash
python scripts/export.py \
  --model runs/train/exp/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --dynamic
```

### 6) Benchmark

```bash
python scripts/benchmark.py \
  --model runs/train/exp/weights/best.pt \
  --data coco8.yaml \
  --imgsz 640 \
  --half \
  --flash-mode turing
```

## What the scripts print

Each script prints:

- selected action
- flash env flags
- resolved backend (`fallback`, `flash_attn`, or `flash_attn_turing`)
- final kwargs passed to Ultralytics

This makes runs easy to audit and reproduce.
