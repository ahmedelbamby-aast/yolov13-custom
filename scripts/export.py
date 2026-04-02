#!/usr/bin/env python3
"""Export YOLOv13 models to deployment formats."""

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
    parser = argparse.ArgumentParser(description="Export a YOLOv13 model.")
    parser.add_argument("--model", required=True, help="Model config or checkpoint path.")
    parser.add_argument("--format", default="onnx", help="Export format: onnx, engine, torchscript, etc.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export graph.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for export graph.")
    parser.add_argument("--device", default="0", help="Export device.")
    parser.add_argument("--half", action="store_true", help="Use FP16 export where supported.")
    parser.add_argument("--int8", action="store_true", help="Use INT8 export where supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes when supported.")
    add_flash_mode_arg(parser)
    add_extra_overrides_arg(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_flash_mode(args.flash_mode)

    YOLO, flash_backend = load_yolo_and_flash_backend()
    model = YOLO(args.model)

    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
    }
    kwargs.update(parse_extra_overrides(args.arg))
    print_runtime_header("export", flash_backend, kwargs)

    artifact = model.export(**kwargs)
    print(f"[y13] export complete. artifact={artifact}")


if __name__ == "__main__":
    main()
