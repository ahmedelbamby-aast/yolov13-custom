#!/usr/bin/env python3
"""API-style benchmark script using YOLO.benchmark()."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import apply_flash_mode, print_runtime, resolve_flash_backend


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark with YOLO API-style calls.")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="coco8.yaml")
    p.add_argument("--task", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--format", default="onnx", help="Single format for API-style benchmark path.")
    p.add_argument("--out-json", default=None)
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
        "imgsz": args.imgsz,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "format": args.format,
    }
    print_runtime("benchmark", args.flash_mode, backend, kwargs)

    bench = model.benchmark(**kwargs)
    print(f"[api-style] benchmark complete. type={type(bench)}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "flash_backend": backend,
            "kwargs": kwargs,
            "benchmark": str(bench),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[api-style] wrote benchmark summary to {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[api-style] benchmark failed: {e}", file=sys.stderr)
        raise
