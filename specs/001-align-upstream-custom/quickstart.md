# Quickstart: Run Alignment Planning Gates

## 1) Confirm baseline and scope

1. Confirm the baseline reference for this cycle:
   - Upstream repo: `https://github.com/ultralytics/ultralytics`
   - Baseline commit: `d93fb45e033d9447fe58918b1265687d8aabb0b5`
   - Nearest tag: `v8.4.38`
2. Confirm scope includes all tasks, modes, and auxiliary developer workflows.

## 2) Build parity inventory

1. Enumerate in-scope workflows across Python API, CLI modes, task flows, and auxiliary tools.
2. Record each workflow with status: `aligned`, `intentionally_different`, or `pending_alignment`.

## 3) Validate compatibility and custom preservation

1. Execute compatibility gates for in-scope upstream workflows.
2. Execute regression gates for release-blocking custom features.
3. Aggregate machine-readable outputs and human-readable summaries.

## 4) Handle intentional exceptions

1. For each intentional difference, create/update exception record with required fields:
   owner, rationale, risk, rollback/mitigation, remediation date.
2. Verify auto-approval conditions are satisfied before marking exception approved.

## 5) Produce release evidence package

1. Consolidate parity inventory, gate outputs, and exception registry.
2. Set release decision:
   - `approved` only if all release-blocking gates pass and exception records are complete.
   - otherwise `blocked`.

## 6) Documentation sync

1. Update `README.md` and `scripts/README.md` to reflect any user-facing behavior changes.
2. Publish migration notes for upstream users and existing fork users.
