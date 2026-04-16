# Phase 0 Research: Upstream Alignment with Custom Feature Preservation

## Decision 1: Upstream baseline anchoring model

- **Decision**: Use `upstream/main` commit `d93fb45e033d9447fe58918b1265687d8aabb0b5`
  (nearest tag `v8.4.38`) as the immutable alignment baseline for this wave.
- **Rationale**: A pinned baseline makes drift measurable, auditable, and repeatable across
  implementation, testing, and release decisions.
- **Alternatives considered**:
  - Floating baseline at latest upstream head each run (rejected: non-deterministic planning).
  - Tag-only baseline without commit pinning (rejected: insufficient precision for audits).

## Decision 2: Scope of parity in this wave

- **Decision**: Include all core upstream user workflows in one wave: Python API, CLI modes,
  task families, and auxiliary developer workflows used in standard operation.
- **Rationale**: The feature spec defines full-wave parity scope and reducing scope would violate
  expected user outcomes.
- **Alternatives considered**:
  - Incremental mode-by-mode rollout (rejected for this feature because scope is explicit).
  - API-only parity without scripts/tooling parity (rejected: incomplete user alignment).

## Decision 3: Compatibility contract posture

- **Decision**: Preserve upstream call semantics by default, including import path and model
  entry behavior, while allowing additive namespaced extensions for custom capabilities.
- **Rationale**: This matches constitution principles I and II and minimizes migration cost for
  upstream users.
- **Alternatives considered**:
  - Fork-specific API namespace (rejected: increases adoption friction).
  - Silent replacement of upstream behavior (rejected: violates constitution and increases risk).

## Decision 4: Gate architecture for release readiness

- **Decision**: Require dual-gate approval: (a) upstream compatibility gates and
  (b) custom-feature regression gates. Any critical gate failure blocks release.
- **Rationale**: Alignment without preserving custom behavior fails business intent; preserving
  custom behavior without parity fails adoption intent.
- **Alternatives considered**:
  - Compatibility-only gates (rejected: can regress custom value).
  - Custom-only gates (rejected: can drift from upstream contract).

## Decision 5: Exception handling policy

- **Decision**: Intentional parity exceptions are allowed only with complete record fields:
  owner, rationale, risk, rollback/mitigation, and remediation date; auto-approval applies only
  when mandatory parity and custom-regression gates pass.
- **Rationale**: Enables controlled deviations without eroding governance quality.
- **Alternatives considered**:
  - Manual approvals without evidence thresholds (rejected: subjective and inconsistent).
  - Disallow all exceptions (rejected: impractical for custom-fork sustainability).

## Decision 6: Documentation synchronization strategy

- **Decision**: Treat `README.md`, `scripts/README.md`, and release migration notes as mandatory
  deliverables for any user-facing behavior change.
- **Rationale**: Documentation parity is required to preserve user trust and reduce support load.
- **Alternatives considered**:
  - Docs update only at major releases (rejected: creates stale guidance windows).
  - Internal notes only (rejected: does not support external consumers).

## Decision 7: Artifact traceability model

- **Decision**: Persist parity inventories, gate outputs, exception registers, and release
  evidence packages under tracked repository paths per cycle.
- **Rationale**: Traceability is required by constitution principle V and simplifies audits.
- **Alternatives considered**:
  - Ephemeral CI logs only (rejected: poor historical auditability).
  - Human summaries without machine-readable evidence (rejected: not verifiable).
