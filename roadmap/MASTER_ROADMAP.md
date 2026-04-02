# YOLOv13 Multi-Task Master Roadmap

## Objective

Enable production-grade YOLOv13 support for:

- Pose Estimation
- Instance Segmentation
- Oriented Bounding Boxes (OBB)

in the custom fork while preserving detect-task stability.

## Current-State Summary

- Both upstream and custom fork already include Ultralytics multi-task plumbing:
  - task map entries for `segment`, `pose`, `obb`
  - task trainers/validators/predictors
  - task model classes (`SegmentationModel`, `PoseModel`, `OBBModel`)
  - dataset pipeline flags for masks/keypoints/obb
- YOLOv13 family is currently practical detect-first in model configs.
- Missing piece: v13-native task model configs and fully validated release path.
- Known issue cluster from community indicates docs/config/metrics friction for OBB/Pose/Seg.

## Architecture Direction

Keep YOLOv13 backbone + neck and add task-specific heads using established Ultralytics conventions.

```mermaid
flowchart LR
    A[YOLOv13 Backbone] --> B[YOLOv13 Neck/FullPAD]
    B --> C1[Detect Head]
    B --> C2[Segment Head]
    B --> C3[Pose Head]
    B --> C4[OBB Head]
```

## Required Deliverables

### 1) Model Configs

- Base task configs:
  - `ultralytics/cfg/models/v13/yolov13-seg.yaml`
  - `ultralytics/cfg/models/v13/yolov13-pose.yaml`
  - `ultralytics/cfg/models/v13/yolov13-obb.yaml`
- Optional scale wrappers per task (`n/s/l/x`) for consistent UX.

### 2) Correctness Fixes

- Align OBB metrics key/value outputs in `ultralytics/utils/metrics.py`.
- Add task-specific preflight validation and clear failure messages.

Status:

- OBB metrics key alignment: in progress (Phase 1 step implemented).

### 3) Data Contracts

- Seg: polygons/masks schema and empty-object behavior.
- Pose: `kpt_shape`, visibility conventions, multi-instance labeling.
- OBB: angle convention (`xywhr`), conversion requirements (e.g., DOTA-style).

### 4) Validation + QA Matrix

- Smoke train/val/predict for each task:
  - Seg: `coco8-seg`
  - Pose: `coco8-pose`
  - OBB: `dota8`
- DDP smoke on 2xT4 for all tasks.
- Export checks (`onnx`, `engine`) with support notes.

### 5) Release Assets

- Repro scripts + pinned env matrix.
- Metrics/artifacts bundle (plots, logs, checkpoints).
- Updated docs and troubleshooting mapped to real issue patterns.

## Phased Execution Plan

```mermaid
flowchart TD
    A[Phase 1: Compatibility fixes] --> B[Phase 2: v13 task YAMLs]
    B --> C[Phase 3: data contracts + validators]
    C --> D[Phase 4: smoke + DDP matrix]
    D --> E[Phase 5: export and runtime checks]
    E --> F[Phase 6: docs, benchmark, release]
```

```mermaid
gantt
    title YOLOv13 Multi-Task Enablement
    dateFormat  YYYY-MM-DD
    section Foundation
    Compatibility fixes            :a1, 2026-04-03, 2d
    section Models
    Add v13 task configs           :a2, after a1, 3d
    section Data
    Data contract tooling          :a3, after a2, 2d
    section Validation
    Smoke and DDP matrix           :a4, after a3, 4d
    section Release
    Docs, benchmark, release       :a5, after a4, 2d
```

## Specs to Author

- `SPEC-10-v13-task-heads.md`
- `SPEC-11-data-contracts.md`
- `SPEC-12-validation-matrix.md`
- `SPEC-13-release-gates.md`

## Knowledge Prerequisites

- Ultralytics task internals and trainer lifecycle.
- Dataset conversion and annotation QA for seg/pose/obb.
- DDP behavior and mixed precision on T4.
- Export constraints per task.

## Issue-to-Action Mapping

```mermaid
flowchart TB
    I1[OBB issue cluster] --> A1[v13-obb configs]
    I1 --> A2[OBB metrics alignment]
    I2[Seg issue cluster] --> A3[v13-seg configs]
    I2 --> A4[segment data preflight]
    I3[Pose issue cluster] --> A5[v13-pose configs]
    I3 --> A6[keypoint schema preflight]
    I4[Environment instability] --> A7[version matrix + diagnostics]
```

## Acceptance Criteria

- [ ] v13 seg/pose/obb configs load and train in smoke mode.
- [ ] Detect baseline remains stable.
- [ ] DDP smoke passes on 2xT4 for all tasks.
- [ ] OBB/pose/seg metrics and callbacks are consistent.
- [ ] Predict/val/export paths are validated and documented.
- [ ] Docs include quickstarts, dataset format, troubleshooting, and limits.

## Folder Index

- `roadmap/01_assessment.md`
- `roadmap/02_architecture_plan.md`
- `roadmap/03_execution_plan.md`
- `roadmap/04_specs.md`
- `roadmap/05_issue_mapping.md`
- `roadmap/06_acceptance_checklists.md`
