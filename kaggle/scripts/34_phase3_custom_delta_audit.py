#!/usr/bin/env python3
"""Audit that key custom deltas remain intact after upstream-alignment changes."""

from __future__ import annotations

import json
from pathlib import Path


def has_text(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except Exception:
        return False


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    checks = {
        "version_8_4_33": has_text(repo / "ultralytics" / "__init__.py", '__version__ = "8.4.33"'),
        "trainer_task_preflight": has_text(
            repo / "ultralytics" / "engine" / "trainer.py", "check_det_dataset(self.args.data, task=self.args.task)"
        ),
        "validator_task_preflight": has_text(
            repo / "ultralytics" / "engine" / "validator.py", "check_det_dataset(self.args.data, task=self.args.task)"
        ),
        "world_task_preflight": has_text(
            repo / "ultralytics" / "models" / "yolo" / "world" / "train_world.py",
            "check_det_dataset(self.args.data, task=self.args.task)",
        ),
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

    out = {
        "total": len(checks),
        "ok": sum(1 for v in checks.values() if v),
        "fail": sum(1 for v in checks.values() if not v),
        "checks": checks,
    }

    out_path = Path("/kaggle/working/phase3_custom_delta_audit.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
