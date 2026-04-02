# Spec 03 - Automation and Packaging

## Goal
Provide a modular one-command flow for setup, checks, smoke tests, and project packaging.

## Implementation Plan
1. Build task scripts in `kaggle/scripts`:
   - `00_entry_banner.py`
   - `10_setup_uv.sh`
   - `20_install_deps.sh`
   - `30_gpu_check.sh`
   - `40_ddp_smoke.sh`
   - `50_package_zip.sh`
2. Add orchestrator `run_all.sh` with strict ordering.
3. Generate output archive at `/kaggle/working/yolov13.zip`.
4. Keep all artifacts and docs synchronized with git and remote repo.

## Dependency Diagram
```text
run_all.sh
  -> 00_entry_banner.py
  -> 10_setup_uv.sh
  -> 20_install_deps.sh
  -> 30_gpu_check.sh
  -> 40_ddp_smoke.sh
  -> 50_package_zip.sh
```

## Detailed Task List
- [x] Script modularization.
- [x] Installer with dependency edge-case handling.
- [x] Add colorful Rich entry banner with owner attribution.
- [x] Orchestrator script.
- [x] Quickstart guide.
- [x] End-to-end pipeline execution report.

## Edge Cases
- Incompatible wheel entries (e.g., local flash-attn filename).
- Optional dependency install failure should not block core path.
- Archive should avoid VCS and venv heavy files.
