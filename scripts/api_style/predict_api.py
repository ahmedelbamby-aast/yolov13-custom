#!/usr/bin/env python3
"""API-style prediction script similar to `results = model(source)` usage."""

from __future__ import annotations

import argparse
import sys

from common import apply_flash_mode, print_runtime, resolve_flash_backend


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict with YOLO API-style calls.")
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="0")
    p.add_argument("--save", action="store_true")
    p.add_argument("--project", default="/kaggle/working/runs/predict")
    p.add_argument("--name", default="api_style_predict")
    p.add_argument("--show", action="store_true")
    p.add_argument("--flash-mode", choices=("auto", "fallback", "turing"), default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    apply_flash_mode(args.flash_mode)

    from ultralytics import YOLO

    backend = resolve_flash_backend()
    model = YOLO(args.model, task=args.task) if args.task else YOLO(args.model)

    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "save": args.save,
        "project": args.project,
        "name": args.name,
        "show": args.show,
    }
    print_runtime("predict", args.flash_mode, backend, kwargs)

    results = model.predict(**kwargs)
    print(f"[api-style] predict complete. results={len(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[api-style] predict failed: {e}", file=sys.stderr)
        raise
