#!/usr/bin/env python3
"""Audit that key custom deltas remain intact after upstream-alignment changes."""

from __future__ import annotations

import json
from pathlib import Path

from phase3_upgrade.common_artifacts import (
    load_baseline,
    load_parity_exceptions,
    load_release_evidence,
    save_parity_exceptions,
    save_release_evidence,
    save_gate_json,
    validate_exception_records,
)


def has_text(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except Exception:
        return False


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    world_train_text = (repo / "ultralytics" / "models" / "yolo" / "world" / "train_world.py").read_text(
        encoding="utf-8"
    )

    version_is_8_4_38 = has_text(repo / "ultralytics" / "__init__.py", '__version__ = "8.4.38"')
    checks = {
        "version_8_4_38": version_is_8_4_38,
        "version_8_4_37": has_text(repo / "ultralytics" / "__init__.py", '__version__ = "8.4.37"') or version_is_8_4_38,
        "trainer_task_preflight": has_text(
            repo / "ultralytics" / "engine" / "trainer.py", "check_det_dataset(self.args.data, task=self.args.task)"
        ),
        "validator_task_preflight": has_text(
            repo / "ultralytics" / "engine" / "validator.py", "check_det_dataset(self.args.data, task=self.args.task)"
        ),
        "world_task_preflight": "check_det_dataset(" in world_train_text and "task=self.args.task" in world_train_text,
        "metrics_has_map75": has_text(repo / "ultralytics" / "utils" / "metrics.py", '"metrics/mAP75(B)"'),
        "block_flash_configure": has_text(
            repo / "ultralytics" / "nn" / "modules" / "block.py", "configure_flash_backend"
        ),
        "block_flash_configured_flag": has_text(
            repo / "ultralytics" / "nn" / "modules" / "block.py", "FLASH_CONFIGURED"
        ),
        "dist_flash_env_propagation": has_text(
            repo / "ultralytics" / "utils" / "dist.py", 'os.environ["Y13_USE_TURING_FLASH"]'
        )
        and has_text(repo / "ultralytics" / "utils" / "dist.py", 'os.environ["Y13_DISABLE_FLASH"]'),
        "trainer_musgd_present": has_text(repo / "ultralytics" / "engine" / "trainer.py", '"MuSGD"'),
    }

    config_files = [
        "yolov13-seg.yaml",
        "yolov13-pose.yaml",
        "yolov13-obb.yaml",
        "yolov13n-seg.yaml",
        "yolov13s-seg.yaml",
        "yolov13l-seg.yaml",
        "yolov13x-seg.yaml",
        "yolov13n-pose.yaml",
        "yolov13s-pose.yaml",
        "yolov13l-pose.yaml",
        "yolov13x-pose.yaml",
        "yolov13n-obb.yaml",
        "yolov13s-obb.yaml",
        "yolov13l-obb.yaml",
        "yolov13x-obb.yaml",
    ]
    cfg_root = repo / "ultralytics" / "cfg" / "models" / "v13"
    checks["v13_task_config_set_present"] = all((cfg_root / name).exists() for name in config_files)

    script_files = ["train.py", "val.py", "test.py", "predict.py", "export.py", "benchmark.py"]
    script_root = repo / "scripts"
    checks["developer_scripts_present"] = all((script_root / name).exists() for name in script_files)

    baseline = load_baseline()
    exceptions_doc = load_parity_exceptions()
    exception_validation = validate_exception_records(exceptions_doc.get("exceptions", []))
    checks["baseline_ref_present"] = bool(baseline.get("resolved_commit"))
    checks["exceptions_records_complete"] = exception_validation["incomplete"] == 0

    exceptions_doc.setdefault("exceptions", [])
    if not exceptions_doc["exceptions"]:
        exceptions_doc["exceptions"].append(
            {
                "exception_id": "E-PLACEHOLDER-001",
                "workflow_id": "placeholder-workflow",
                "rationale": "No approved parity exceptions in this cycle",
                "risk_level": "low",
                "owner": "y13-maintainers",
                "rollback_or_mitigation": "No-op",
                "remediation_date": "2026-12-31",
                "approval_state": "rejected",
            }
        )
        save_parity_exceptions(exceptions_doc)

    out = {
        "total": len(checks),
        "ok": sum(1 for v in checks.values() if v),
        "fail": sum(1 for v in checks.values() if not v),
        "checks": checks,
        "baseline_ref": baseline.get("resolved_commit"),
        "exceptions": exception_validation,
    }

    out_path = save_gate_json("phase3_custom_delta_audit.json", out)

    evidence = load_release_evidence()
    gates = evidence.setdefault("gates", {})
    custom_regression = gates.setdefault("custom_regression", {})
    custom_regression["status"] = "pass" if out["fail"] == 0 else "fail"
    refs = custom_regression.setdefault("evidence_refs", [])
    refs.append("phase3_custom_delta_audit.json")
    save_release_evidence(evidence)

    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
