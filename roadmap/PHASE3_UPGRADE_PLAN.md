# Phase 3: Ultralytics Upgrade and Dependency Alignment

## Goal

Upgrade this custom fork from `ultralytics==8.3.63` toward the latest upstream release while preserving:

- YOLOv13 detect/segment/pose/obb support
- custom task preflight checks
- OBB metric compatibility updates
- flash backend controls (fallback/turing)
- Kaggle and developer script workflows

## Branch

- Working branch: `phase3-upgrade-ultralytics-and-deps`

## Current status

- Started Phase 3 branch and pushed first upgrade commit.
- Added Muon-family optimizer support (`MuSGD` + `Muon`) and integrated optimizer selection into trainer.
- Verified with a real 1-epoch Kaggle run using `optimizer=MuSGD`.
- Upgraded package version surface to `ultralytics==8.4.33` in `ultralytics/__init__.py`.
- Synced project metadata and dependency baselines in `pyproject.toml` to latest upstream format.
- Refreshed `requirements.txt` runtime pins for current fork workflows.

## Upgrade workstreams

1. Upstream code sync
   - Compare upstream latest `ultralytics` core against fork.
   - Merge low-risk core updates module-by-module.
2. Optimizer and training parity
   - Keep `MuSGD`/`optimizer=auto` behavior aligned with upstream.
   - Validate single-GPU and DDP behavior.
3. Dependency refresh
   - Align pinned runtime dependencies with latest stable compatible matrix.
   - Verify PyTorch/CUDA/TensorRT/ONNX paths in Kaggle.
4. Compatibility retention
   - Re-verify YOLOv13 config load matrix.
   - Re-run preflight smoke tests for detect/segment/pose/obb.
5. Benchmarks and regression checks
   - Smoke train/val/export/benchmark on updated stack.
   - Re-check fallback vs turing backend reporting.

## Exit criteria

- All core workflows pass on Kaggle and local dev scripts.
- No regressions in YOLOv13 task behavior.
- Dependency set documented and reproducible.
- Phase branch ready for merge into `main`.
