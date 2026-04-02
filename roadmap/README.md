# YOLOv13 Multi-Task Roadmap

This folder contains the implementation roadmap and supporting artifacts to make YOLOv13 fully production-ready for:

- Pose Estimation
- Instance Segmentation
- OBB (Oriented Bounding Boxes)

## Contents

- `01_assessment.md` - Current-state assessment of upstream vs custom fork.
- `02_architecture_plan.md` - Architecture strategy and task-head design.
- `03_execution_plan.md` - Phased roadmap with milestones and timeline.
- `04_specs.md` - Proposed specification documents and required decisions.
- `05_issue_mapping.md` - Mapping open issues to fixes and validation.
- `06_acceptance_checklists.md` - Delivery checklist for engineering and QA.
- `MASTER_ROADMAP.md` - Consolidated all-in-one roadmap.
- `specs/` - Detailed specs for implementation and release gates.

## Specs Included

- `specs/SPEC-10-v13-task-heads.md`
- `specs/SPEC-11-data-contracts.md`
- `specs/SPEC-12-validation-matrix.md`
- `specs/SPEC-13-release-gates.md`

## Scope

This roadmap focuses on bringing the YOLOv13 model family from detect-only practical usage to full multi-task support with reproducible training, validation, export, and documentation.
