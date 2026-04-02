# Phase 2: Custom Feature Replay on Upstream-Aligned Core

## Objective

Reapply and validate all fork-specific functional deltas on top of the upstream-aligned baseline from Phase 1.

## Scope

- YOLOv13 configs and task heads.
- Task-aware dataset preflight validation.
- OBB metric compatibility behavior.
- Turing flash backend control and deterministic selection path.
- DDP env propagation for backend flags.

## Tasks

1. YOLOv13 model/config layer
   - Validate `ultralytics/cfg/models/v13/*` mapping and compatibility with latest code paths.
   - Ensure `detect/segment/pose/obb` task variants load across `n/s/l/x`.

2. Data preflight layer
   - Re-validate `_sample_label_files` and `_validate_task_label_schema` behavior.
   - Ensure trainer/validator/world pass task context correctly.

3. Metrics layer
   - Confirm OBB metrics keys include required `mAP75(B)` behavior.

4. Flash backend layer
   - Keep fallback/turing controls deterministic.
   - Keep runtime re-config path and DDP override propagation valid.

## Acceptance Criteria

- All v13 task configs load for safe scales.
- Preflight smoke checks pass for detect/segment/pose/obb.
- OBB metrics compatibility retained.
- Turing vs fallback backend mode is selectable and reported correctly.
