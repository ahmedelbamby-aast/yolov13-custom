#!/usr/bin/env python3
"""API-style test script (validation over configurable split)."""

from __future__ import annotations

import argparse
import sys

from common import apply_flash_mode, print_runtime, resolve_flash_backend


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test with YOLO API-style calls.")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--task", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--flash-mode", choices=("auto", "fallback", "turing"), default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    apply_flash_mode(args.flash_mode)

    from ultralytics import YOLO

    backend = resolve_flash_backend()
    model = YOLO(args.model, task=args.task) if args.task else YOLO(args.model)

    kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
    }
    print_runtime("test", args.flash_mode, backend, kwargs)

    metrics = model.val(**kwargs)
    summary = getattr(metrics, "results_dict", None)
    print(f"[api-style] test complete. metrics={summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[api-style] test failed: {e}", file=sys.stderr)
        raise
