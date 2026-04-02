#!/usr/bin/env python3
"""Train YOLOv13 models with a simple developer-facing interface."""

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
    parser = argparse.ArgumentParser(description="Train a YOLOv13 model.")
    parser.add_argument("--model", required=True, help="Model config or checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Device string, e.g. '0', '0,1', 'cpu'.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="exp", help="Run name.")
    parser.add_argument("--task", default=None, help="Optional task override (detect/segment/pose/obb).")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    parser.add_argument("--amp", type=bool, default=True, help="Use AMP.")
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
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "amp": args.amp,
    }
    if args.task:
        kwargs["task"] = args.task

    kwargs.update(parse_extra_overrides(args.arg))
    print_runtime_header("train", flash_backend, kwargs)

    model.train(**kwargs)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    print(f"[y13] training complete. save_dir={save_dir}")


if __name__ == "__main__":
    main()
