# SPEC-12: Validation Matrix

## Purpose

Standardize task-level verification before merge/release.

## Test Levels

1. Unit/Static checks
   - YAML parse checks
   - metrics key integrity checks

2. Smoke checks (1 epoch)
   - Segment on `coco8-seg`
   - Pose on `coco8-pose`
   - OBB on `dota8`

3. Functional checks (10 epochs)
   - Verify decreasing loss and non-empty outputs

4. Runtime checks
   - `train`, `val`, `predict`
   - DDP smoke on 2xT4

5. Export checks
   - ONNX for each task where supported
   - TensorRT where supported and documented

Phase 1 harness:

- `kaggle/scripts/33_phase1_task_preflight_smoke.py`
  - validates task preflight behavior on synthetic valid/invalid labels for segment/pose/obb.

## Matrix

| Task | Train | Val | Predict | DDP | Export |
|---|---|---|---|---|---|
| Detect | Required | Required | Required | Required | Required |
| Segment | Required | Required | Required | Required | Required |
| Pose | Required | Required | Required | Required | Required |
| OBB | Required | Required | Required | Required | Required |

## Artifacts Required

- `results.csv`
- `args.yaml`
- `run_summary.json`
- metric plots
- sample predictions
- (optional) feature maps

## Acceptance

- Every matrix cell passes for release candidate branch.
- Any task failure blocks release unless explicitly waived.
