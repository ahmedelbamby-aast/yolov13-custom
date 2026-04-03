#!/usr/bin/env python3
"""API-style export script similar to direct YOLO.export usage."""

from __future__ import annotations

import argparse
import sys

from common import apply_flash_mode, print_runtime, resolve_flash_backend


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export with YOLO API-style calls.")
    p.add_argument("--model", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--flash-mode", choices=("auto", "fallback", "turing"), default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    apply_flash_mode(args.flash_mode)

    from ultralytics import YOLO

    backend = resolve_flash_backend()
    model = YOLO(args.model, task=args.task) if args.task else YOLO(args.model)

    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
    }
    print_runtime("export", args.flash_mode, backend, kwargs)

    artifact = model.export(**kwargs)
    print(f"[api-style] export complete. artifact={artifact}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[api-style] export failed: {e}", file=sys.stderr)
        raise
