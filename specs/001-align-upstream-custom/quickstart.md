# Quickstart: Run Alignment Planning Gates

## 1) Confirm baseline and scope

1. Confirm the baseline reference for this cycle:
   - Upstream repo: `https://github.com/ultralytics/ultralytics`
   - Baseline commit: `d93fb45e033d9447fe58918b1265687d8aabb0b5`
   - Nearest tag: `v8.4.38`
2. Confirm scope includes all tasks, modes, and auxiliary developer workflows.

## 1.5) Run required gate order

Run the implementation gates in this strict order:

1. `kaggle/scripts/phase3_upgrade/00_alignment_schema_check.py`
2. `kaggle/scripts/phase3_upgrade/02_cli_python_parity_gate.py`
3. `kaggle/scripts/phase3_upgrade/03_stress_gate.py`
4. `kaggle/scripts/34_phase3_custom_delta_audit.py`
5. `kaggle/scripts/36_phase3_final_gate.py`

## 1.6) Remote progress visibility (mandatory)

1. For any long-running remote workflow, expose progress in one of these ways:
   - live log stream (`tail -f`/equivalent), or
   - periodic status heartbeat at least every 5 minutes.
2. Save progress evidence paths (logs/status artifacts) with other gate outputs.

## 1.7) Server safety constraints (mandatory)

1. Do not execute reboot/shutdown/system-file deletion operations unless:
   - the developer has been informed, and
   - explicit approval has been recorded.
2. If such an action is required, keep release state `blocked` until authorization evidence is attached.

## 1.8) Virtual environment and platform profile

1. Run bootstrap scripts that create and activate a dedicated virtual environment before installs/gates.
2. Record host runtime profile for each cycle:
   - OS family (Windows/macOS/Linux)
   - execution style (headless/interactive)
   - accelerator profile (CPU-only/single-GPU/multi-GPU)
3. Keep setup behavior compatible with non-Kaggle hosts using the same documented steps.

## 1.9) Flash backend auto-selection and validation

1. Auto-detect hardware capability and attempt supported flash backend installation/testing.
2. Prefer Flash Tur (`flash-attention-turing`) on T4-compatible environments when validation passes.
3. Preserve deterministic fallback behavior when flash install/validation is unsupported or fails.

## 2) Build parity inventory

1. Enumerate in-scope workflows across Python API, CLI modes, task flows, and auxiliary tools.
2. Record each workflow with status: `aligned`, `intentionally_different`, or `pending_alignment`.

## 3) Validate compatibility and custom preservation

1. Execute compatibility gates for in-scope upstream workflows.
   - include `kaggle/scripts/33_phase1_task_preflight_smoke.py` and `kaggle/scripts/35_phase_ddp_gate.py` as custom-regression prerequisites for release evidence.
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

3. Verify SC-001 metric from parity artifacts:
   - `ratio = aligned WorkflowParityItems / in-scope WorkflowParityItems`
   - approved intentional differences are excluded from denominator.

## 5.5) Canonical publication gate (release-blocking)

1. Push approved updates to `https://github.com/ahmedelbamby-aast/yolov13-custom`.
2. Record publication evidence (commit hash, branch, timestamp) in `release-evidence.yaml`.
3. Keep release decision `blocked` until publication push is successful.

## 6) Documentation sync

1. Update `README.md` and `scripts/README.md` to reflect any user-facing behavior changes.
2. Publish migration notes for upstream users and existing fork users.
