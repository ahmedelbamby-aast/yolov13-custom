# 04 Specs

## Proposed Spec Documents

1. `SPEC-10-v13-task-heads.md`
   - Defines architecture deltas for seg/pose/obb on top of v13.
   - Includes head layer wiring, tensor shape expectations, and scale handling.

2. `SPEC-11-data-contracts.md`
   - Segment label format and polygon constraints.
   - Pose `kpt_shape`, visibility conventions, multi-person rules.
   - OBB xywhr conventions and dataset conversion requirements.

3. `SPEC-12-validation-matrix.md`
   - Required smoke tests, DDP tests, and export checks.
   - Pass/fail criteria and artifact requirements.

4. `SPEC-13-release-gates.md`
   - Defines blockers for release.
   - Includes backward-compat checks and docs completeness gates.

## Decision Log (to resolve early)

- Head design source of truth:
  - Minimal adaptation from v8/v11 task heads, or deeper v13-specific head redesign.
- OBB metrics output format:
  - Include mAP75 in official key list or keep compact output and adapt callbacks.
- Checkpoint naming and task-specific export defaults.
- Minimum supported torch/cuda matrix for production.
