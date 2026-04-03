# Phase 4: Validation and Benchmark Parity

## Objective

Prove that the upgraded stack preserves expected behavior and performance characteristics across workflows.

## Scope

- Functional validation across train/val/test/export/predict.
- Benchmark comparisons for fallback vs turing.
- L-scale multi-task benchmark coverage.

## Tasks

1. Functional matrix
   - Train/val/test/predict/export smoke for detect/segment/pose/obb.
   - Verify developer scripts with mode-arg passthrough.

2. Benchmark matrix
   - Re-run fallback vs turing benchmark suites.
   - Regenerate comparison plots and reports.

3. Regression checks
   - Compare critical metrics and runtime trends versus pre-upgrade baseline.

## Acceptance Criteria

- All key workflows pass on Kaggle.
- Plot/report artifacts are regenerated and synced.
- No critical regressions in task functionality.

## Progress Snapshot

- Re-ran L-scale fallback vs turing benchmark suite on upgraded branch (5 epochs):
  - output root: `/kaggle/working/phase3_l_flash_compare_5e`
  - fallback backend: `fallback`
  - turing backend: `flash_attn_turing`
  - tasks covered: `detect`, `segment`, `pose`, `obb`
- Synced benchmark artifacts and report:
  - `kaggle/benchmarks/l_flash_tasks/`
  - `kaggle/reports/BENCHMARK_L_FLASH_TASKS_COMPARISON.md`
- Added benchmark summary artifact:
  - `roadmap/artifacts/phase3_l_flash_compare_5e_summary.json`
