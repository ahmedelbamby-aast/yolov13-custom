<!--
Sync Impact Report
- Version change: 1.1.0 -> 1.1.1
- Modified principles:
  - II. Additive and Namespaced Customization -> II. Additive and Namespaced Customization
- Added sections:
  - None
- Removed sections:
  - None
- Templates requiring updates:
  - ⚠ pending: .specify/templates/plan-template.md
  - ⚠ pending: .specify/templates/spec-template.md
  - ⚠ pending: .specify/templates/tasks-template.md
  - ⚠ pending: .specify/templates/commands/*.md (directory not present)
  - ⚠ pending: README.md and scripts/README.md normalization notes
- Follow-up TODOs:
  - Normalize flash-mode vocabulary across script families.
  - Add fork-specific tests for Y13 backend, preflight schema, and DDP hardening.
-->

# YOLOv13 Custom Fork Constitution

## Core Principles

### I. Upstream Contract Preservation
Public Python and CLI contracts MUST remain upstream-compatible by default, including
`from ultralytics import YOLO`, model loading semantics, and task/mode argument behavior.
Any intentional contract difference MUST be documented as a compatibility exception with
impact, migration note, and remediation target. This rule protects drop-in usability.

### II. Additive and Namespaced Customization
Fork features MUST be additive and explicitly scoped (for example, v13 model configs,
`Y13_*` runtime controls, and script-level flags). Shared upstream behavior MUST NOT be
silently replaced unless required for correctness, security, or hard reliability defects.
Custom features that are currently supported by this fork MUST NOT be deleted during
upstream alignment. Any proposal to remove a custom feature requires an explicit deprecation
decision, migration path, and owner-approved exception record. All replacements MUST include
rationale and rollback guidance.

### III. Fail-Fast Data and Runtime Determinism
Dataset/task contracts MUST be validated before heavy execution (especially detect/segment/
pose/obb schema differences). Runtime backend selection MUST be deterministic and applied
before or at controlled model initialization boundaries. The system MUST fail early with
actionable diagnostics when schema or backend constraints are violated.

### IV. Reliability for Distributed and Long-Running Training
Distributed and long-running execution paths MUST prioritize stability over novelty: safe
synchronization, non-finite guardrails, controlled restart/resume behavior, and explicit
failure containment. DDP, checkpointing, and runtime fallback behavior MUST remain robust
under partial failure and constrained hardware conditions.

### V. Evidence-Based Quality and Release Gates
Claims of parity, stability, or performance MUST be backed by executable gates and
machine-readable artifacts (logs, metrics, summaries, and status JSON). Changes to entry
points, train/val/predict/export/benchmark flows, or backend control MUST pass import smoke,
task-relevant integration checks, and no-regression script checks before release.

## Engineering Constitutes

- Baseline Declaration Clause: each parity initiative MUST pin upstream repository and commit/
  tag reference in plan artifacts.
- Delta Accounting Clause: each meaningful change MUST be classified as parity sync, custom
  feature, reliability fix, tooling, or docs-only.
- Compatibility Gate Clause: any change affecting entrypoints or core flows MUST pass defined
  smoke/integration/regression checks before merge.
- Exception Clause: skipped gates or intentional parity breaks MUST include owner, risk,
  rollback strategy, and remediation date.
- Artifact Clause: gate and benchmark outputs MUST be persisted in traceable locations.
- Documentation Sync Clause: behavior changes MUST update user docs and script guidance.

## Operating Practices

- Scripts SHOULD remain thin wrappers over native Ultralytics API calls with full argument
  passthrough rather than parallel logic stacks.
- Benchmarking SHOULD use controlled export formats, explicit backend labels, and
  machine-readable summaries to support apples-to-apples comparisons.
- Kaggle and remote bootstrap workflows SHOULD be deterministic and reproducible across runs.
- Roadmap/spec artifacts SHOULD remain evidence-linked to executed gates and produced outputs.
- New custom features SHOULD ship with targeted tests, not only manual gate scripts.

## Governance

- This constitution governs architecture, API surface, reliability expectations, and release
  readiness decisions for this repository.
- Amendments require: proposed text, rationale, impact assessment, dependent-template review,
  and Sync Impact Report updates in this file.
- Constitution versioning uses semantic rules:
  - MAJOR: incompatible redefinition/removal of core principles or governance rules.
  - MINOR: added principles/sections or materially expanded mandatory requirements.
  - PATCH: clarifications, editorial refinements, and non-semantic wording updates.
- Compliance checks are mandatory for plans and implementation reviews affecting APIs,
  data contracts, runtime backends, DDP behavior, or release gates.
- Non-compliance MUST be logged as a named exception with owner, risk, and due date.
- Runtime and developer guidance in `README.md`, `scripts/README.md`, and `roadmap/` MUST
  remain aligned with this constitution.

**Version**: 1.1.1 | **Ratified**: 2026-04-16 | **Last Amended**: 2026-04-16
