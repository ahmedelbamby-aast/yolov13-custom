# Implementation Plan: Upstream Alignment with Custom Feature Preservation

**Branch**: `003-align-upstream-custom` | **Date**: 2026-04-16 | **Spec**: `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\spec.md`
**Input**: Feature specification from `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\spec.md`

**Note**: This plan follows `.specify/templates/plan-template.md` and the active constitution.

## Summary

Deliver fork-wide behavioral parity with upstream Ultralytics for all core workflows while
preserving existing YOLOv13 custom capabilities. The plan uses an auditable parity baseline,
explicit exception governance, and release gates that require both upstream-compatibility
success and custom-regression success before approval.

## Technical Context

**Language/Version**: Python >=3.8 (project metadata), validated on supported Python matrix  
**Primary Dependencies**: torch, torchvision, numpy, opencv-python, pyyaml, requests, scipy, polars, ultralytics-thop  
**Storage**: File-based artifacts (weights, logs, reports, JSON gate outputs, markdown docs)  
**Testing**: pytest test suite (`tests/`), CLI/API parity gate scripts (`kaggle/scripts/phase3_upgrade/`), script-level smoke suites  
**Target Platform**: Linux, Windows, macOS; GPU-enabled environments for DDP and flash-backend validation  
**Project Type**: Python library + CLI + automation scripts  
**Performance Goals**: Maintain upstream-equivalent functional behavior while preserving benchmark reproducibility and existing fork reliability gates  
**Constraints**: No public API contract breaks by default; custom features must be additive and namespaced; release blocked on failing critical parity/regression checks  
**Scale/Scope**: All upstream tasks and modes plus auxiliary developer workflows for one alignment wave

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Upstream baseline is recorded: `https://github.com/ultralytics/ultralytics`, ref `upstream/main` at
  `d93fb45e033d9447fe58918b1265687d8aabb0b5` (nearest tag `v8.4.38`).
- API parity impact is documented for top-level import and constructor behavior, including
  `from ultralytics import YOLO` and `YOLO("...")` mode/task flows.
- Design is additive-first: custom behavior remains namespaced and must not silently replace
  upstream behavior without a justified exception.
- Compatibility and regression gates are defined: import smoke, task/mode parity checks,
  script no-regression checks, and custom-feature regression gates.
- Documentation update plan is included for `README.md`, `scripts/README.md`, and release
  migration notes when behavior changes.
- No justified violations at plan time. Constitution gate status: PASS (pre-research).

## Project Structure

### Documentation (this feature)

```text
C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── parity-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
C:\Users\Ahmed\yolov13_custom\ultralytics\
├── __init__.py
├── cfg/
├── engine/
├── models/
└── utils/

C:\Users\Ahmed\yolov13_custom\scripts\
├── train.py
├── val.py
├── test.py
├── predict.py
├── export.py
├── benchmark.py
├── _common.py
└── api_style/

C:\Users\Ahmed\yolov13_custom\kaggle\scripts\
├── phase3_upgrade/
└── *.sh/*.py gate and benchmark scripts

C:\Users\Ahmed\yolov13_custom\tests\
├── test_cli.py
├── test_python.py
├── test_engine.py
├── test_exports.py
└── ...
```

**Structure Decision**: Single Python library/CLI repository with workflow scripts and gate
automation; implementation will modify existing modules and validation scripts rather than
introducing new top-level projects.

## Phase 0: Outline & Research

Research tracks produced from technical context and spec scope:

1. Upstream baseline strategy and drift accounting model for all tasks/modes.
2. Public API compatibility matrix and no-break policy for import/constructor usage.
3. Best-practice release gates combining upstream parity and custom regression checks.
4. Exception policy model for intentional parity differences.

Output artifact: `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\research.md`

## Phase 1: Design & Contracts

Design outputs derived from `spec.md` and `research.md`:

1. Data model capturing parity baseline, custom registry, delta records, and release evidence.
2. Interface contract documenting required user-facing compatibility and exception semantics.
3. Quickstart defining operator workflow for running parity + custom gates and assembling
   release readiness evidence.
4. Agent context refresh for OpenCode to reflect planning artifacts and governance constraints.

Output artifacts:

- `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\data-model.md`
- `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\contracts\parity-contract.md`
- `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\quickstart.md`

Post-design constitution check status: PASS (no unresolved clarifications, no unjustified gate violation).

## Complexity Tracking

No constitution violations require justification at planning phase.
