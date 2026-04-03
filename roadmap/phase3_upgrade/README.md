# Phase 3 Upgrade Roadmap

Goal: reach parity with latest Ultralytics while preserving all custom YOLOv13 work in this fork.

Branch:

- `phase3-upgrade-ultralytics-and-deps`

Phases:

1. `PHASE1_UPSTREAM_BASELINE_SYNC.md`
   - Supporting artifacts:
     - `PHASE1_DIFF_BASELINE.md`
     - `PHASE1_CORE_PARITY_MAP.md`
2. `PHASE2_CUSTOM_FEATURE_REPLAY.md`
3. `PHASE3_DEPENDENCY_RUNTIME_ALIGNMENT.md`
4. `PHASE4_VALIDATION_BENCHMARK_PARITY.md`
5. `PHASE5_RELEASE_AND_MERGE.md`

Execution principles:

- Keep each phase mergeable and testable.
- Push incremental commits after each milestone.
- Preserve custom behavior with explicit acceptance checks.
- Prefer upstream behavior by default, then layer custom deltas with minimal drift.
