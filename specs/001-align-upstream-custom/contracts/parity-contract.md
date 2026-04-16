# Contract: Upstream Parity and Custom Preservation

## Purpose

Define the externally observable behavior contract that the fork must satisfy to be considered
aligned with upstream while preserving custom features.

## Consumer Roles

- Developer consuming Python API.
- Developer consuming CLI/mode workflows.
- Maintainer validating parity and release readiness.

## Behavioral Contract

### 1) Public Entry Contract

- The fork MUST support upstream-style primary imports and model loading semantics for all
  in-scope workflows.
- Existing upstream-style usage flows MUST remain valid unless explicitly listed as exceptions.

### 2) Workflow Outcome Contract

- For each in-scope upstream workflow, the fork MUST classify status as:
  - `aligned`
  - `intentionally_different`
  - `pending_alignment`
- Release approval MUST require all release-blocking workflows to be `aligned` or approved
  `intentionally_different`.

### 3) Custom Feature Preservation Contract

- Every custom feature in the release-blocking registry MUST remain available and behaviorally
  valid in release candidates.
- Any removal or degradation requires explicit deprecation handling and approved exception.

### 4) Exception Contract

- Intentional divergence from upstream MUST include:
  - owner,
  - rationale,
  - risk,
  - rollback/mitigation,
  - remediation date.
- Exceptions MAY be auto-approved only when all mandatory parity and custom-regression gates
  pass and record fields are complete.

### 5) Release Decision Contract

- Release decision MUST be `blocked` when any release-blocking gate fails.
- Release decision MAY be `approved` only with complete evidence package containing parity,
  custom regression, and exception documentation.
- Release decision MUST remain `blocked` until canonical publication push succeeds to
  `https://github.com/ahmedelbamby-aast/yolov13-custom`.

### 6) SC-001 Measurement Contract

- SC-001 MUST be computed as:
  - `aligned WorkflowParityItems / in-scope WorkflowParityItems`
- Approved intentional differences MUST be excluded from the denominator.
- Gate outputs MUST publish numerator, denominator, and ratio as machine-readable values.

## Required Evidence Artifacts

- Parity inventory snapshot.
- Compatibility gate outputs.
- Custom regression gate outputs.
- Exception registry snapshot.
- Final release evidence package with decision.

## Non-Goals

- This contract does not require internal code to be structurally identical to upstream.
- This contract does not mandate immediate remediation for every non-release-blocking drift item.
