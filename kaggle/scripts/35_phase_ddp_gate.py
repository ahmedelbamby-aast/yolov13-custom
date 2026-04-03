#!/usr/bin/env python3
"""Run a standardized DDP smoke gate and write JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


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

    out_path = Path(f"/kaggle/working/{args.phase}_ddp_gate.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
