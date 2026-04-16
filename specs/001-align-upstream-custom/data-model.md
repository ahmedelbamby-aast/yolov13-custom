# Data Model: Upstream Alignment with Custom Feature Preservation

## Entity: UpstreamBaseline

- **Purpose**: Defines immutable upstream reference for a parity cycle.
- **Fields**:
  - `repo_url` (string, required)
  - `reference_type` (enum: `branch` | `tag` | `commit`, required)
  - `reference_value` (string, required)
  - `resolved_commit` (string, required)
  - `captured_at` (datetime, required)
  - `scope_definition` (text, required)
- **Validation Rules**:
  - `resolved_commit` must be immutable hash format.
  - `scope_definition` must enumerate tasks, modes, and auxiliary workflows.

## Entity: WorkflowParityItem

- **Purpose**: Represents parity state of one in-scope workflow.
- **Fields**:
  - `workflow_id` (string, required, unique within cycle)
  - `category` (enum: `python_api` | `cli_mode` | `task_flow` | `aux_tooling`, required)
  - `status` (enum: `aligned` | `intentionally_different` | `pending_alignment`, required)
  - `evidence_refs` (list[string], required)
  - `notes` (text, optional)
- **Validation Rules**:
  - `intentionally_different` requires linked `ParityException`.
  - `aligned` requires at least one compatibility evidence reference.

## Entity: CustomFeatureRegistryItem

- **Purpose**: Declares a custom feature that must be preserved.
- **Fields**:
  - `feature_id` (string, required, unique)
  - `name` (string, required)
  - `criticality` (enum: `release_blocking` | `important` | `optional`, required)
  - `expected_outcome` (text, required)
  - `regression_test_refs` (list[string], required)
  - `owner` (string, required)
- **Validation Rules**:
  - `release_blocking` items require passing regression evidence for release approval.

## Entity: ParityException

- **Purpose**: Controlled record of intentional divergence from upstream behavior.
- **Fields**:
  - `exception_id` (string, required, unique)
  - `workflow_id` (string, required, FK -> WorkflowParityItem.workflow_id)
  - `rationale` (text, required)
  - `risk_level` (enum: `low` | `medium` | `high`, required)
  - `owner` (string, required)
  - `rollback_or_mitigation` (text, required)
  - `remediation_date` (date, required)
  - `approval_state` (enum: `pending` | `auto_approved` | `manually_approved` | `rejected`, required)
- **Validation Rules**:
  - `auto_approved` allowed only when mandatory parity and custom gates pass.
  - `remediation_date` must be future or same-day at approval time.

## Entity: GateRun

- **Purpose**: Captures one execution of compatibility or regression gates.
- **Fields**:
  - `gate_run_id` (string, required, unique)
  - `gate_type` (enum: `compatibility` | `custom_regression` | `release_summary`, required)
  - `started_at` (datetime, required)
  - `ended_at` (datetime, required)
  - `result` (enum: `pass` | `fail` | `partial`, required)
  - `artifact_paths` (list[string], required)
  - `triggered_by` (string, required)
- **Validation Rules**:
  - `release_summary` requires upstream and custom gate evidence links.
  - Any `fail` on release-blocking gates prevents release approval.

## Entity: ReleaseEvidencePackage

- **Purpose**: Consolidated release readiness record for one candidate.
- **Fields**:
  - `release_id` (string, required, unique)
  - `baseline_ref` (string, required, FK -> UpstreamBaseline.resolved_commit)
  - `parity_snapshot_ref` (string, required)
  - `custom_snapshot_ref` (string, required)
  - `exceptions_ref` (string, required)
  - `decision` (enum: `approved` | `blocked`, required)
  - `decision_reason` (text, required)
  - `published_at` (datetime, optional)
- **Validation Rules**:
  - `approved` requires no failing release-blocking gate.
  - `approved` requires complete exception records for all intentional differences.

## Relationships

- `UpstreamBaseline` 1 -> N `WorkflowParityItem`
- `WorkflowParityItem` 0 -> 1 `ParityException`
- `CustomFeatureRegistryItem` N -> N `GateRun` (via evidence references)
- `GateRun` N -> 1 `ReleaseEvidencePackage`
- `ParityException` N -> 1 `ReleaseEvidencePackage`

## State Transitions

### WorkflowParityItem.status

- `pending_alignment` -> `aligned` when compatibility evidence passes.
- `pending_alignment` -> `intentionally_different` when approved exception exists.
- `intentionally_different` -> `aligned` when remediation completes and tests pass.

### ReleaseEvidencePackage.decision

- `blocked` (default candidate state)
- `blocked` -> `approved` only after:
  - compatibility gates pass,
  - release-blocking custom regressions pass,
  - exceptions are complete and approved.
