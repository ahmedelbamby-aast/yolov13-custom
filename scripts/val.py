#!/usr/bin/env python3
"""Validate YOLOv13 models on custom datasets."""

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
    parser = argparse.ArgumentParser(description="Validate a YOLOv13 model.")
    parser.add_argument("--model", required=True, help="Model config or checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Device string, e.g. '0', '0,1', 'cpu'.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--split", default="val", help="Dataset split: val/test/train.")
    parser.add_argument("--task", default=None, help="Optional task override (detect/segment/pose/obb).")
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
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "split": args.split,
    }
    if args.task:
        kwargs["task"] = args.task

    kwargs.update(parse_extra_overrides(args.arg))
    print_runtime_header("val", flash_backend, kwargs)

    metrics = model.val(**kwargs)
    print(f"[y13] validation complete. metrics={metrics}")


if __name__ == "__main__":
    main()
