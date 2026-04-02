#!/usr/bin/env python3
"""Run YOLOv13 inference on images/video/dirs."""

from __future__ import annotations

import argparse
import sys

from _common import (
    add_extra_overrides_arg,
    add_flash_mode_arg,
    apply_flash_mode,
    load_yolo_and_flash_backend,
    merge_kwarg_sources,
    parse_unknown_cli_overrides,
    parse_extra_overrides,
    print_runtime_header,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv13 prediction.")
    parser.add_argument("--model", required=True, help="Model config or checkpoint path.")
    parser.add_argument("--source", required=True, help="Source path/url/webcam id.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default="0", help="Device string, e.g. '0', 'cpu'.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--project", default="runs/predict", help="Output project directory.")
    parser.add_argument("--name", default="exp", help="Run name.")
    parser.add_argument("--save", action="store_true", help="Save predictions.")
    parser.add_argument("--stream", action="store_true", help="Use streaming inference.")
    add_flash_mode_arg(parser)
    add_extra_overrides_arg(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    apply_flash_mode(args.flash_mode)

    YOLO, flash_backend = load_yolo_and_flash_backend()
    model = YOLO(args.model)

    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.project,
        "name": args.name,
        "save": args.save,
        "stream": args.stream,
    }
    extra_overrides = parse_extra_overrides(args.arg)
    unknown_overrides = parse_unknown_cli_overrides(unknown)
    kwargs = merge_kwarg_sources(kwargs, extra_overrides, unknown_overrides)
    print_runtime_header("predict", flash_backend, kwargs)

    results = model.predict(**kwargs)
    count = "stream" if args.stream else len(results)
    print(f"[y13] predict complete. results={count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[y13] predict failed: {e}", file=sys.stderr)
        raise
