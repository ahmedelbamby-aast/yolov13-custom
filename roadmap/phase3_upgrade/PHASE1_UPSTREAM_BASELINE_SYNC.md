# Phase 1: Upstream Baseline Sync

## Objective

Create a stable baseline that matches latest Ultralytics core patterns as closely as possible, before replaying custom fork features.

## Scope

- Align package/version/metadata surfaces with upstream latest.
- Align foundational training optimizer behavior with upstream (`MuSGD`, auto selection path).
- Establish a measurable diff baseline against `upstream/main`.
- Do not yet migrate all custom scripts/artifacts; keep focus on core runtime baseline.

## Tasks

1. Upstream reference wiring
   - Ensure `upstream` remote points to `https://github.com/ultralytics/ultralytics.git`.
   - Capture upstream version and critical file references.

2. Core metadata and init parity
   - Align `ultralytics/__init__.py` structure and version surface.
   - Align `pyproject.toml` metadata/dependency format.

3. Optimizer parity baseline
   - Introduce/align `ultralytics/optim/muon.py` and `ultralytics/optim/__init__.py`.
   - Align trainer `build_optimizer()` behavior for `MuSGD` and `optimizer=auto` path.

4. Runtime verification
   - Smoke import on Kaggle.
   - One-epoch training probe with `optimizer=MuSGD`.

5. Diff baseline report
   - Generate file-diff summary versus upstream.
   - Identify high-risk modules to defer to later phases.

## Acceptance Criteria

- `from ultralytics import YOLO` imports successfully on Kaggle.
- `optimizer=MuSGD` runs a 1-epoch training probe without optimizer errors.
- Phase branch contains an explicit baseline diff snapshot and plan for next phases.

## Current Progress

- In progress and partially complete.
- `MuSGD` support integrated and verified by real Kaggle run.
- `__init__` and `pyproject.toml` baseline alignment completed.
