#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ultralytics.data.utils import check_det_dataset


@dataclass
class CaseResult:
    name: str
    task: str
    expected: str
    status: str
    detail: str


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(127, 127, 127)).save(path)


def _write_label(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line.rstrip() + "\n", encoding="utf-8")


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _build_case(root: Path, name: str, task: str, label_line: str, pose_kpt_shape: str | None = None) -> Path:
    case = root / name
    for split in ("train", "val"):
        _write_image(case / split / "images" / "im0.jpg")
        _write_label(case / split / "labels" / "im0.txt", label_line)

    extra = f"kpt_shape: {pose_kpt_shape}\n" if pose_kpt_shape else ""
    yaml_text = f"""
path: {case}
train: train/images
val: val/images
nc: 1
names:
  0: cls0
{extra}
"""
    yaml_path = case / "data.yaml"
    _write_yaml(yaml_path, yaml_text)
    return yaml_path


def run() -> dict:
    out_root = Path("/kaggle/working/phase1_preflight_smoke")
    out_root.mkdir(parents=True, exist_ok=True)

    cases: list[CaseResult] = []

    # Valid cases
    valid_segment = _build_case(
        out_root,
        "segment_valid",
        "segment",
        "0 0.10 0.10 0.30 0.10 0.30 0.30",
    )
    valid_pose = _build_case(
        out_root,
        "pose_valid",
        "pose",
        "0 0.50 0.50 0.40 0.40 0.10 0.10 0.20 0.20 0.30 0.30",
        pose_kpt_shape="[3, 2]",
    )
    valid_obb = _build_case(
        out_root,
        "obb_valid",
        "obb",
        "0 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30",
    )

    # Invalid cases
    invalid_segment = _build_case(
        out_root,
        "segment_invalid",
        "segment",
        "0 0.10 0.10 0.30 0.10",  # too short
    )
    invalid_pose = _build_case(
        out_root,
        "pose_invalid",
        "pose",
        "0 0.50 0.50 0.40 0.40 0.10 0.10",  # token count mismatch for [3,2]
        pose_kpt_shape="[3, 2]",
    )
    invalid_obb = _build_case(
        out_root,
        "obb_invalid",
        "obb",
        "0 0.10 0.10 0.30 0.10 0.30 0.30",  # too short for corners
    )

    matrix = [
        ("segment_valid", "segment", "pass", valid_segment),
        ("pose_valid", "pose", "pass", valid_pose),
        ("obb_valid", "obb", "pass", valid_obb),
        ("segment_invalid", "segment", "fail", invalid_segment),
        ("pose_invalid", "pose", "fail", invalid_pose),
        ("obb_invalid", "obb", "fail", invalid_obb),
    ]

    for name, task, expected, yaml_path in matrix:
        try:
            check_det_dataset(str(yaml_path), autodownload=False, task=task)
            status = "pass"
            detail = "validated"
        except Exception as e:  # noqa: BLE001
            status = "fail"
            detail = str(e)

        ok = (expected == "pass" and status == "pass") or (expected == "fail" and status == "fail")
        cases.append(
            CaseResult(
                name=name,
                task=task,
                expected=expected,
                status="ok" if ok else "mismatch",
                detail=detail,
            )
        )

    summary = {
        "total": len(cases),
        "ok": sum(1 for c in cases if c.status == "ok"),
        "mismatch": sum(1 for c in cases if c.status != "ok"),
        "cases": [c.__dict__ for c in cases],
    }

    report_json = out_root / "phase1_task_preflight_smoke.json"
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"report_json={report_json}")
    return summary


if __name__ == "__main__":
    run()
