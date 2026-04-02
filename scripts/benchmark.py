#!/usr/bin/env python3
"""Benchmark YOLOv13 models across runtime/export settings."""

from __future__ import annotations

import argparse

from _common import (
    add_extra_overrides_arg,
    add_flash_mode_arg,
    apply_flash_mode,
    load_yolo_and_flash_backend,
    parse_extra_overrides,
    print_runtime_header,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a YOLOv13 model.")
    parser.add_argument("--model", required=True, help="Model config or checkpoint path.")
    parser.add_argument("--data", default=None, help="Optional dataset yaml for benchmark.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default="0", help="Device for benchmark.")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported.")
    parser.add_argument("--int8", action="store_true", help="Use INT8 where supported.")
    parser.add_argument("--verbose", action="store_true", help="Verbose benchmark output.")
    add_flash_mode_arg(parser)
    add_extra_overrides_arg(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_flash_mode(args.flash_mode)

    YOLO, flash_backend = load_yolo_and_flash_backend()
    model = YOLO(args.model)

    kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "verbose": args.verbose,
    }
    kwargs.update(parse_extra_overrides(args.arg))
    print_runtime_header("benchmark", flash_backend, kwargs)

    out = model.benchmark(**kwargs)
    print(f"[y13] benchmark complete. output={out}")


if __name__ == "__main__":
    main()
