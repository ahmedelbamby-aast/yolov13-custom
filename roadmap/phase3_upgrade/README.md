# Phase 3 Upgrade Roadmap

Goal: reach parity with latest Ultralytics while preserving all custom YOLOv13 work in this fork.

Branch:

- `phase3-upgrade-ultralytics-and-deps`

Phases:

1. `PHASE1_UPSTREAM_BASELINE_SYNC.md`
   - Supporting artifacts:
     - `PHASE1_DIFF_BASELINE.md`
     - `PHASE1_CORE_PARITY_MAP.md`
     - `PHASE1_VERIFICATION.md`
2. `PHASE2_CUSTOM_FEATURE_REPLAY.md`
3. `PHASE3_DEPENDENCY_RUNTIME_ALIGNMENT.md`
4. `PHASE4_VALIDATION_BENCHMARK_PARITY.md`
5. `PHASE5_RELEASE_AND_MERGE.md`

Execution principles:

- Keep each phase mergeable and testable.
- Push incremental commits after each milestone.
- Preserve custom behavior with explicit acceptance checks.
- Prefer upstream behavior by default, then layer custom deltas with minimal drift.
- Mandatory gate: run at least one DDP smoke validation at the end of each phase.

Recommended gate command:

- `python kaggle/scripts/35_phase_ddp_gate.py --phase <phase-tag> --flash-mode auto`

Current status:

- Phase 1 verification gates are passing with artifact evidence committed.
- Phase 3 dependency/runtime alignment has successful first execution snapshot on Kaggle.
- Phase 2 replay smoke matrix is passing for fallback+turing across detect/segment/pose/obb.
- Phase 4 benchmark rerun on upgraded baseline completed for fallback vs turing (`detect/segment/pose/obb`, 5e).
- Standardized per-phase DDP gate script is added and validated on Phase 3.
- T4 GPU benchmark blocker is resolved with controlled benchmark runner (`scripts/benchmark.py`).
- Phase3 final integration gate is passing end-to-end (train/val/predict/export/benchmark).
- Next execution step: release merge from `phase3-upgrade-ultralytics-and-deps` to `main`.
