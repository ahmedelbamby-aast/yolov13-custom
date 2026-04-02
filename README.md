<p align="center">
  <img src="assets/icon.png" width="110" alt="YOLOv13" />
</p>

<h2 align="center">YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception</h2>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.17733)
[![iMoonLab](https://img.shields.io/badge/iMoonLab-Homepage-blueviolet?logo=github&logoColor=white)](https://github.com/iMoonLab)
[![Fork](https://img.shields.io/badge/Fork-ahmedelbamby--aast/yolov13--custom-1f6feb?logo=github)](https://github.com/ahmedelbamby-aast/yolov13-custom)

<div align="center">
  <img src="assets/framework.png" alt="YOLOv13 framework" />
</div>

This repository is a production-focused custom fork of YOLOv13. It keeps upstream model work while adding reliability, reproducibility, multi-task support, and developer tooling for real-world training/validation/export/benchmark workflows.

## Table of Contents

- [Overview](#overview)
- [Update Timeline](#update-timeline)
- [What This Fork Adds](#what-this-fork-adds)
- [Quick Links](#quick-links)
- [Installation](#installation)
- [Usage](#usage)
- [Flash Backend Control](#flash-backend-control)
- [Benchmarks and Reports](#benchmarks-and-reports)
- [Repository Layout](#repository-layout)
- [Related Projects](#related-projects)
- [Citation](#citation)
- [Known Runtime Warning](#known-runtime-warning)

## Overview

YOLOv13 introduces HyperACE + FullPAD for stronger feature interaction and robust real-time detection. This fork extends the base project with:

- YOLOv13 multi-task configs (detect, segment, pose, obb)
- task-aware dataset preflight validation
- deterministic flash backend control (fallback vs turing)
- modular developer scripts aligned with Ultralytics mode usage
- Kaggle-ready automation for 2x T4 DDP workflows

## Update Timeline

### Upstream milestones (iMoonLab)

- 2025-06-21: initial YOLOv13 code open-sourced
- 2025-06-22: YOLOv13 weights released
- 2025-06-24: paper and deployment integrations published
- 2025-06-25 to 2025-11-18: API/demo/community deployment updates

### Fork milestones (ahmedelbamby-aast)

- 2026-04: phase-1 reliability updates (OBB metric compatibility + task preflight validation)
- 2026-04: phase-2 YOLOv13 task-head configs for segment/pose/obb (`n/s/l/x`)
- 2026-04: deterministic flash backend re-init and DDP env propagation
- 2026-04: L-scale fallback vs turing benchmark suite with comparison plots
- 2026-04: modular `scripts/` workflows for train/val/test/predict/export/benchmark with full mode-arg passthrough

## What This Fork Adds

- **Multi-task model configs** under `ultralytics/cfg/models/v13/` for detect/segment/pose/obb variants.
- **Task-aware data preflight** in dataset checks and trainer/validator/world entry points.
- **Flash backend control** with deterministic selection (`auto`, `fallback`, `turing`) and Turing support path.
- **Developer scripts** that follow normal Ultralytics usage while supporting all mode arguments.
- **Kaggle automation** for setup, GPU checks, DDP smoke, packaging, and benchmark reproducibility.

## Quick Links

- Full environment and run guide: `kaggle/QUICKSTART.md`
- Developer scripts guide: `scripts/README.md`
- L-task benchmark report: `kaggle/reports/BENCHMARK_L_FLASH_TASKS_COMPARISON.md`
- S/L/X benchmark reports:
  - `kaggle/reports/BENCHMARK_SLX_30E.md`
  - `kaggle/reports/BENCHMARK_SLX_30E_FIXED_BATCH_COMPARISON.md`

## Installation

### Local/Server (standard)

```bash
git clone https://github.com/ahmedelbamby-aast/yolov13-custom.git
cd yolov13-custom
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

### Kaggle (recommended path)

Follow `kaggle/QUICKSTART.md` for full bootstrap, GPU setup, and pipelines.

## Usage

You can use either the native Ultralytics API or the modular scripts.

### Native Ultralytics API

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v13/yolov13l.yaml")
model.train(data="coco8.yaml", epochs=100, imgsz=640)
model.val(data="coco8.yaml")
model.predict(source="path/to/images", save=True)
model.export(format="onnx")
```

### Modular scripts (dev-friendly)

```bash
python scripts/train.py --model ultralytics/cfg/models/v13/yolov13l.yaml --data coco8.yaml --epochs 100
python scripts/val.py --model runs/train/exp/weights/best.pt --data coco8.yaml
python scripts/test.py --model runs/train/exp/weights/best.pt --data coco8.yaml --split test
python scripts/predict.py --model runs/train/exp/weights/best.pt --source path/to/images --save
python scripts/export.py --model runs/train/exp/weights/best.pt --format onnx
python scripts/benchmark.py --model runs/train/exp/weights/best.pt --data coco8.yaml --half
```

These scripts support all Ultralytics mode arguments by passthrough:

- direct style: `--optimizer AdamW --lr0 0.001 --patience 50`
- key/value style: `--arg optimizer=AdamW --arg lr0=0.001`

Reference modes: https://docs.ultralytics.com/modes/

## Flash Backend Control

### Per-script toggle

All scripts accept:

- `--flash-mode auto`
- `--flash-mode fallback`
- `--flash-mode turing`

### Environment toggle

```bash
export Y13_USE_TURING_FLASH=1
export Y13_DISABLE_FLASH=0
# export Y13_DISABLE_FLASH=1  # force fallback
```

## Benchmarks and Reports

- L-scale task comparison (detect/segment/pose/obb, fallback vs turing):
  - script: `kaggle/scripts/181_benchmark_l_flash_tasks.sh`
  - artifacts: `kaggle/benchmarks/l_flash_tasks/`
  - report: `kaggle/reports/BENCHMARK_L_FLASH_TASKS_COMPARISON.md`

- S/L/X comparison suites:
  - scripts: `kaggle/scripts/160_benchmark_slx_optimal_batch.py`, `kaggle/scripts/170_benchmark_slx_fixed_batch.py`
  - reports under `kaggle/reports/`

## Repository Layout

- `ultralytics/`: model code, configs, training/validation/export logic
- `scripts/`: modular developer workflows
- `kaggle/scripts/`: automation and validation pipelines
- `kaggle/reports/`: benchmark and implementation reports
- `kaggle/benchmarks/`: benchmark metrics and plots
- `roadmap/`: specs, plans, and implementation tracking

## Related Projects

- Base framework: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Upstream YOLOv13: [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)
- Hypergraph references:
  - [Hypergraph Neural Networks](https://arxiv.org/abs/1809.09401)
  - [HGNN+](https://ieeexplore.ieee.org/abstract/document/9795251)
  - [SoftHGNN](https://arxiv.org/abs/2505.15325)

## Citation

```bibtex
@article{yolov13,
  title={YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception},
  author={Lei, Mengqi and Li, Siqi and Wu, Yihong and et al.},
  journal={arXiv preprint arXiv:2506.17733},
  year={2025}
}
```

## Known Runtime Warning

You may still see repeated logs such as:

- `Unable to register cuDNN factory`
- `Unable to register cuBLAS factory`
- `computation placer already registered`

These are usually environment-level CUDA/XLA/TensorFlow-linked warnings and are non-fatal for PyTorch training in this repo.
