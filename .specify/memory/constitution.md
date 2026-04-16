<!--
Sync Impact Report
- Version change: N/A (template placeholders) -> 1.0.0
- Modified principles:
  - Template Principle 1 -> I. Upstream API Parity First
  - Template Principle 2 -> II. Additive Customization Only
  - Template Principle 3 -> III. Evidence-Driven Upstream Synchronization
  - Template Principle 4 -> IV. Compatibility and Regression Gates
  - Template Principle 5 -> V. Developer Experience Consistency
- Added sections:
  - API and Architecture Constraints
  - Delivery Workflow and Review Gates
- Removed sections:
  - None
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
  - ⚠ pending: .specify/templates/commands/*.md (directory not present)
  - ✅ README.md reviewed (no amendment needed)
- Follow-up TODOs:
  - None
-->

# YOLOv13 Custom Fork Constitution

## Core Principles

### I. Upstream API Parity First
All public APIs exposed by this fork MUST preserve upstream Ultralytics call patterns and
behavior by default, including import paths and primary constructors such as
`from ultralytics import YOLO` and `YOLO("model")`. Any divergence from upstream semantics
MUST be additive, explicitly documented, and protected by compatibility tests. This keeps
developer onboarding friction low and prevents ecosystem breakage.

### II. Additive Customization Only
YOLOv13 custom features MUST be implemented as additive extensions, not replacements of
upstream behavior, unless replacement is required to fix correctness or security defects.
Custom capabilities MUST use explicit namespacing and configuration boundaries (for example,
v13 configs and `Y13_*` environment flags) so upstream-compatible workflows remain intact.
This preserves mergeability with upstream and avoids hidden coupling.

### III. Evidence-Driven Upstream Synchronization
Every upstream sync or parity change MUST include a documented delta analysis that covers
API surface, key module patterns, and behavioral differences against the chosen upstream
reference (commit or release tag). Unresolved drifts MUST be tracked with owner and target
milestone. This ensures parity work is measurable and auditable rather than ad hoc.

### IV. Compatibility and Regression Gates
Changes affecting entry points, trainers, validators, predictors, exporters, or model loading
MUST pass compatibility gates before merge: import smoke tests, task-relevant integration
tests (detect/segment/pose/obb when affected), and no-regression checks for existing scripts.
If a gate is skipped, a written exception with risk and rollback plan is mandatory.

### V. Developer Experience Consistency
Documentation, examples, and scripts MUST maintain the same API consumption style as
Ultralytics wherever possible, with custom extensions clearly marked and optional.
User-facing guidance MUST include both upstream-equivalent usage and fork-specific usage,
including defaults and migration notes. This keeps the fork easy to adopt for existing
Ultralytics users.

## API and Architecture Constraints

- Package import stability is mandatory: `ultralytics` remains the top-level public module.
- Public API additions MUST be backward-compatible and MUST NOT shadow or alter existing
  upstream signatures without a documented compatibility layer.
- Custom model/config assets MUST reside in clearly scoped locations (for example,
  `ultralytics/cfg/models/v13/`) to reduce conflict during upstream merges.
- Core behavior changes in shared modules MUST include inline rationale in PR/spec artifacts,
  plus explicit reference to upstream source behavior.
- External dependency upgrades MUST be justified with parity, security, or reliability impact.

## Delivery Workflow and Review Gates

1. Establish upstream baseline: record upstream repo URL and reference commit/tag for each
   parity initiative.
2. Produce delta inventory: list changed modules, APIs, and behavior differences in spec/plan.
3. Implement with additive-first strategy and minimal shared-core churn.
4. Validate gates: run compatibility smoke checks and affected task tests before merge.
5. Update developer guidance: refresh README/examples/scripts where behavior or options change.
6. Record outcome: store parity status and remaining drift items in roadmap/spec artifacts.

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

- This constitution supersedes conflicting local process notes for architecture and API parity
  decisions in this repository.
- Amendments require: (a) proposed change text, (b) rationale and impact, (c) template sync
  review across `.specify/templates/`, and (d) update of the Sync Impact Report header.
- Versioning policy for this constitution uses semantic versioning:
  - MAJOR: backward-incompatible principle or governance changes.
  - MINOR: new principle/section or materially expanded mandatory guidance.
  - PATCH: clarifications, wording improvements, and non-semantic edits.
- Compliance review is required in every plan and implementation review that touches API,
  architecture, or developer workflow. Non-compliance MUST be tracked with explicit exception
  owner, risk, and remediation date.
- Operational guidance remains in `README.md` and feature artifacts under `roadmap/` and
  `specs/`; those documents MUST remain consistent with this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-04-16 | **Last Amended**: 2026-04-16
