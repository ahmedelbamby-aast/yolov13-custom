# Phase 6: Full Upgrade Execution Plan (v8.4.37 Target)

## Objective

Upgrade `phase3-upgrade-ultralytics-and-deps` from `8.4.33` baseline to latest Ultralytics patch line (`8.4.37`), preserve custom YOLOv13 behavior, and validate under high-load single-GPU execution.

## Constraints

- Single-GPU validation is mandatory (`device=0`), no DDP during integration steps.
- Always run with maximum available workers on the VM (`workers=os.cpu_count()`).
- Stress settings for validation: `amp=True`, `cache=ram`, high prefetch where supported.
- Keep Python API and CLI behavior aligned with upstream usage patterns.

## Execution Phases

1. Test-first gate scaffold
   - Add deterministic gate scripts for environment, API/CLI parity, and stress stability.
   - Produce machine-readable artifacts (`*.json`) in `/kaggle/working/phase3_upgrade/`.

2. Upstream patch integration
   - Integrate upstream deltas through `v8.4.37` with minimal drift.
   - Prioritize training stability commits (NaN recovery, AP precision, class weighting, AAttn fixes, DDP logging).

3. Runtime validation under load
   - Run stress gate with `amp=True`, `cache=ram`, max workers.
   - Confirm no fatal OOM at validated batch settings.

4. Documentation closure
   - Update roadmap status and quickstart/gates docs with exact commands and acceptance outcomes.

5. Merge readiness and branch merge
   - Merge `phase3-upgrade-ultralytics-and-deps` into `main` only after all gates pass.

## Acceptance Criteria

- Gate scripts pass on fresh Kaggle VM.
- CLI and Python API both exercise train/val/predict/export paths successfully.
- Stress gate completes with stable settings and records chosen max stable batch.
- Docs reflect final tested commands and outcomes.
- Merge to `main` completed.
## Progress Update (2026-04-11)

### Applied upstream stability patches

- 94d9bf55 Fix loss explosion on resume when training on small dataset (ultralytics/utils/torch_utils.py)
- 18f33939 Prevent DDP resource cleanup issues when DDP command generation fails (ultralytics/engine/trainer.py)
- 7eaea3c6 Allow first-epoch checkpoint save even when EMA has NaN/Inf (ultralytics/engine/trainer.py)
- a4813997 Improve AP interpolation sentinel handling in compute_ap (ultralytics/utils/metrics.py)
- 650fca87 Clamp coordinates in crop_mask before cropping (ultralytics/utils/ops.py)

### Manual upstream-alignment adjustments

- Bumped local package version to 8.4.37 in ultralytics/__init__.py.
- Removed legacy duplicate model assignment path in Model.train() so training uses trainer-owned model setup flow.
- Aligned BaseTrainer.setup_model() verbosity to RANK in {-1, 0}.
- Aligned Pose trainer model construction verbosity to verbose and RANK == -1.

### Validation status

- Phase6 gate runner executed successfully after patch set:
  - 01_env_report.py: PASS
  - 02_cli_python_parity_gate.py: PASS
  - 03_stress_gate.py: PASS (device=0, workers=4, prefetch_factor=8, cache=ram, amp=True, classes=[0,9])
  - 04_autobatch_ddp_notes.py: PASS

### Notes on deferred patches

- edc79913 (AAttn non-divisible head-dim fix) was evaluated but deferred in this pass due high-conflict overlap with custom flash-attention logic in ultralytics/nn/modules/block.py.
- AAttn shape-safety port completed manually in ultralytics/nn/modules/block.py and validated with targeted non-divisible-head testcases plus full phase6 gates.

