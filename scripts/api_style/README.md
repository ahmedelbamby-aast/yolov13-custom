# API-Style Scripts (Ultralytics-like Usage)

This folder adds an alternative script set that looks and feels close to direct API usage:

```python
from ultralytics import YOLO

model = YOLO('yolov13n.yaml')
results = model.train(...)
metrics = model.val(...)
pred = model.predict(...)
```

These scripts do not replace existing scripts in `scripts/`; they are additional developer entry points.

## Files

- `scripts/api_style/train_api.py`
- `scripts/api_style/val_api.py`
- `scripts/api_style/test_api.py`
- `scripts/api_style/predict_api.py`
- `scripts/api_style/export_api.py`
- `scripts/api_style/benchmark_api.py`

## Flash backend control

All scripts support:

- `--flash-mode auto`
- `--flash-mode fallback`
- `--flash-mode turing`

They print:

- selected action
- flash env values
- resolved backend
- final kwargs passed to the API call

## Example usage

### Train (API-style)

```bash
python scripts/api_style/train_api.py \
  --model ultralytics/cfg/models/v13/yolov13n.yaml \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --task detect \
  --epochs 600 \
  --batch 256 \
  --imgsz 640 \
  --scale 0.5 \
  --mosaic 1.0 \
  --mixup 0.0 \
  --copy-paste 0.1 \
  --device 0,1,2,3 \
  --feature-projection \
  --feature-projection-script kaggle/scripts/37_feature_map_projection.py \
  --feature-projection-valid-dir /kaggle/work_here/datasets/my_detect/valid/images \
  --feature-projection-out-dir /kaggle/working/runs/train/api_style_train/feature_projection \
  --feature-projection-md-path /kaggle/working/runs/train/api_style_train/ff_maps.md \
  --feature-projection-log-path /kaggle/working/runs/train/api_style_train/feature_projection.log \
  --feature-projection-flash-mode same \
  --flash-mode turing
```

Feature projection behavior:

- runs only after train completes
- uses `<save_dir>/weights/best.pt`
- can reuse train flash mode (`--feature-projection-flash-mode same`) or override it

### Validate

```bash
python scripts/api_style/val_api.py \
  --model /kaggle/working/runs/train/api_style_train/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --split val \
  --device 0 \
  --flash-mode turing
```

### Test

```bash
python scripts/api_style/test_api.py \
  --model /kaggle/working/runs/train/api_style_train/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --device 0 \
  --flash-mode turing
```

### Predict

```bash
python scripts/api_style/predict_api.py \
  --model /kaggle/working/runs/train/api_style_train/weights/best.pt \
  --source /kaggle/work_here/datasets/my_detect/valid/images \
  --device 0 \
  --save \
  --flash-mode turing
```

### Export

```bash
python scripts/api_style/export_api.py \
  --model /kaggle/working/runs/train/api_style_train/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --dynamic \
  --flash-mode turing
```

### Benchmark

```bash
python scripts/api_style/benchmark_api.py \
  --model /kaggle/working/runs/train/api_style_train/weights/best.pt \
  --data /kaggle/work_here/datasets/my_detect/data.yaml \
  --imgsz 640 \
  --device 0 \
  --format onnx \
  --flash-mode turing \
  --out-json /kaggle/working/api_style_bench.json
```

Note:

- This benchmark path intentionally uses `YOLO.benchmark()` to stay API-style.
- For the controlled T4 benchmark workflow used in phase gates, prefer `scripts/benchmark.py`.
