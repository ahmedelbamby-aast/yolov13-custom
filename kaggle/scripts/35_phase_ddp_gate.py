#!/usr/bin/env python3
"""Run a standardized DDP smoke gate and write JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from phase3_upgrade.common_artifacts import load_release_evidence, save_gate_json, save_release_evidence, utc_now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase DDP smoke gate")
    p.add_argument("--phase", default="phase3", help="Phase label to include in artifacts.")
    p.add_argument("--model", default="ultralytics/cfg/models/v13/yolov13.yaml", help="Model config/path.")
    p.add_argument("--data", default="coco8.yaml", help="Dataset yaml.")
    p.add_argument("--epochs", type=int, default=1, help="Smoke epochs.")
    p.add_argument("--imgsz", type=int, default=64, help="Image size.")
    p.add_argument("--batch", type=int, default=8, help="Batch size.")
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    p.add_argument("--device", default="0,1", help="DDP device spec.")
    p.add_argument("--optimizer", default="MuSGD", help="Optimizer.")
    p.add_argument("--flash-mode", choices=("auto", "fallback", "turing"), default="auto", help="Flash mode.")
    p.add_argument("--project", default="/kaggle/working/phase_ddp", help="Project output root.")
    p.add_argument("--name", default="ddp_gate", help="Run name.")
    return p.parse_args()


def apply_flash_mode(mode: str) -> None:
    if mode == "fallback":
        os.environ["Y13_DISABLE_FLASH"] = "1"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
    elif mode == "turing":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "1"
    else:
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ.setdefault("Y13_USE_TURING_FLASH", "0")


def main() -> None:
    args = parse_args()
    apply_flash_mode(args.flash_mode)

    start = time.time()
    status, error = "ok", ""

    try:
        from ultralytics import YOLO

        YOLO(args.model).train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            optimizer=args.optimizer,
        )
    except Exception as e:
        status, error = "fail", str(e)

    out = {
        "phase": args.phase,
        "gate": "ddp_smoke",
        "started_at": utc_now_iso(),
        "status": status,
        "error": error[:1000],
        "elapsed_s": time.time() - start,
        "flash_mode": args.flash_mode,
        "env": {
            "Y13_DISABLE_FLASH": os.getenv("Y13_DISABLE_FLASH", ""),
            "Y13_USE_TURING_FLASH": os.getenv("Y13_USE_TURING_FLASH", ""),
        },
        "run": {
            "model": args.model,
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "optimizer": args.optimizer,
            "project": args.project,
            "name": args.name,
        },
    }
    out["ended_at"] = utc_now_iso()

    out_path = Path(f"/kaggle/working/{args.phase}_ddp_gate.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_gate_json(f"{args.phase}_ddp_gate.json", out)

    evidence = load_release_evidence()
    gates = evidence.setdefault("gates", {})
    custom_regression = gates.setdefault("custom_regression", {})
    custom_regression["status"] = "pass" if status == "ok" else "fail"
    refs = custom_regression.setdefault("evidence_refs", [])
    refs.append(f"{args.phase}_ddp_gate.json")
    if status != "ok":
        evidence["status"] = "blocked"
        blocking = evidence.setdefault("blocking_reasons", [])
        if "failed_ddp_smoke_gate" not in blocking:
            blocking.append("failed_ddp_smoke_gate")
    save_release_evidence(evidence)

    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
