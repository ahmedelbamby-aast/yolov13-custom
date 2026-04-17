<!--
Sync Impact Report
- Version change: 1.1.1 -> 1.2.0
- Modified principles:
  - III. Fail-Fast Data and Runtime Determinism -> III. Fail-Fast Data and Runtime Determinism
  - IV. Reliability for Distributed and Long-Running Training -> IV. Reliability for Distributed and Long-Running Training
  - V. Evidence-Based Quality and Release Gates -> V. Evidence-Based Quality and Release Gates
- Added sections:
  - VI. Server Safety and Change Control
- Removed sections:
  - None
- Templates requiring updates:
  - ⚠ pending: .specify/templates/plan-template.md
  - ⚠ pending: .specify/templates/spec-template.md
  - ⚠ pending: .specify/templates/tasks-template.md
  - ⚠ pending: .specify/templates/commands/*.md (directory not present)
  - ✅ updated: specs/001-align-upstream-custom/spec.md
- Follow-up TODOs:
  - Add remote execution helpers for live logs and 5-minute progress polling.
  - Add platform-aware flash install/detect gates for Windows, macOS, Linux, GPU, and CPU-only hosts.
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
actionable diagnostics when schema or backend constraints are violated. Long-running
remote execution MUST provide operator-visible progress via live logs or periodic status
checks at least once every 5 minutes.

### IV. Reliability for Distributed and Long-Running Training
Distributed and long-running execution paths MUST prioritize stability over novelty: safe
synchronization, non-finite guardrails, controlled restart/resume behavior, and explicit
failure containment. DDP, checkpointing, and runtime fallback behavior MUST remain robust
under partial failure and constrained hardware conditions. Implementations MUST be
platform-adaptive and operate correctly on Windows, macOS, Linux, and headless servers,
including CPU-only, single-GPU, and multi-GPU topologies.

### V. Evidence-Based Quality and Release Gates
Claims of parity, stability, or performance MUST be backed by executable gates and
machine-readable artifacts (logs, metrics, summaries, and status JSON). Changes to entry
points, train/val/predict/export/benchmark flows, or backend control MUST pass import smoke,
task-relevant integration checks, and no-regression script checks before release. Bootstrap
and runtime flows MUST verify virtual environment activation and backend readiness.

### VI. Server Safety and Change Control
The team MUST NOT reboot, shut down, delete system files, or run destructive server-level
operations unless the developer is informed in advance and explicitly grants approval.
Operational safety takes precedence over speed for infrastructure-impacting actions.

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
- Remote Progress Clause: long-running remote jobs MUST expose live logs or automated progress
  checks at a maximum 5-minute interval.
- Server Safety Clause: destructive host operations (reboot/shutdown/system-file deletion) are
  forbidden without explicit developer authorization.
- Platform Adaptation Clause: scripts and runtime behavior MUST adapt to host OS and hardware
  (Windows/macOS/Linux, CPU-only, single-GPU, multi-GPU) without manual code rewrites.
- Flash Automation Clause: the system MUST auto-detect available GPU capability, install/test
  the best supported flash backend, and prefer Flash Tur for T4-compatible environments.
- Environment Activation Clause: bootstrap scripts MUST create a dedicated virtual environment,
  and execution scripts MUST run inside that environment.

## Operating Practices

- Scripts SHOULD remain thin wrappers over native Ultralytics API calls with full argument
  passthrough rather than parallel logic stacks.
- Benchmarking SHOULD use controlled export formats, explicit backend labels, and
  machine-readable summaries to support apples-to-apples comparisons.
- Kaggle and remote bootstrap workflows SHOULD be deterministic and reproducible across runs.
- Roadmap/spec artifacts SHOULD remain evidence-linked to executed gates and produced outputs.
- New custom features SHOULD ship with targeted tests, not only manual gate scripts.
- Remote runs SHOULD write a durable log file (`tee` or equivalent) that can be tailed without
  waiting for job completion.

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
  data contracts, runtime backends, DDP behavior, server safety, or release gates.
- Non-compliance MUST be logged as a named exception with owner, risk, and due date.
- Runtime and developer guidance in `README.md`, `scripts/README.md`, and `roadmap/` MUST
  remain aligned with this constitution.

**Version**: 1.2.0 | **Ratified**: 2026-04-16 | **Last Amended**: 2026-04-17
