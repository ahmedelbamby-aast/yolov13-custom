#!/usr/bin/env python3
"""API-style training script similar to direct Ultralytics usage."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import (
    apply_flash_mode,
    merge_kwarg_sources,
    parse_kv_overrides,
    parse_unknown_cli_overrides,
    print_runtime,
    resolve_flash_backend,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train with YOLO API-style calls.")
    p.add_argument("--model", default="ultralytics/cfg/models/v13/yolov13n.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--task", default="detect")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--time", type=float, default=None, help="Optional hour budget.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="/kaggle/working/runs/train")
    p.add_argument("--name", default="api_style_train")
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--cache", default=None)
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--amp", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=0.1)
    p.add_argument(
        "--feature-projection",
        action="store_true",
        help="Run per-layer feature-map projection after training completes using best.pt.",
    )
    p.add_argument("--feature-projection-script", default="kaggle/scripts/37_feature_map_projection.py")
    p.add_argument("--feature-projection-source", default="")
    p.add_argument(
        "--feature-projection-valid-dir", default="/kaggle/work_here/datasets/roboflow_custom_detect/valid/images"
    )
    p.add_argument("--feature-projection-device", default="0")
    p.add_argument("--feature-projection-imgsz", type=int, default=None)
    p.add_argument("--feature-projection-out-dir", default="")
    p.add_argument("--feature-projection-md-path", default="")
    p.add_argument("--feature-projection-log-path", default="")
    p.add_argument("--feature-projection-dataset-name", default="SBAS-AASTMT-AI-ALAMEIN")
    p.add_argument("--feature-projection-model-name", default="YOLOv13")
    p.add_argument("--feature-projection-variant", default="l")
    p.add_argument(
        "--feature-projection-flash-mode",
        default="same",
        choices=("same", "auto", "fallback", "turing", "flash4"),
    )
    p.add_argument("--feature-projection-wait-seconds", type=int, default=300)
    p.add_argument("--feature-projection-poll-seconds", type=int, default=10)
    p.add_argument("--flash-mode", choices=("auto", "fallback", "turing", "flash4"), default="auto")
    p.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE")
    return p


def _apply_flash_mode_to_env(env: dict[str, str], mode: str) -> None:
    if mode == "fallback":
        env["Y13_DISABLE_FLASH"] = "1"
        env["Y13_USE_TURING_FLASH"] = "0"
        env["Y13_PREFER_FLASH4"] = "0"
    elif mode == "turing":
        env["Y13_DISABLE_FLASH"] = "0"
        env["Y13_USE_TURING_FLASH"] = "1"
        env["Y13_PREFER_FLASH4"] = "0"
    elif mode == "flash4":
        env["Y13_DISABLE_FLASH"] = "0"
        env["Y13_USE_TURING_FLASH"] = "0"
        env["Y13_PREFER_FLASH4"] = "1"
    else:
        env["Y13_DISABLE_FLASH"] = "0"
        env["Y13_USE_TURING_FLASH"] = "0"
        env["Y13_PREFER_FLASH4"] = "0"


def _run_feature_projection(args: argparse.Namespace, save_dir: str | Path | None) -> None:
    if not args.feature_projection:
        return

    if not save_dir:
        raise RuntimeError("feature projection requested but trainer save_dir is unavailable")

    save_dir = Path(save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"feature projection requested but best.pt not found: {best_pt}")

    projection_script = Path(args.feature_projection_script)
    if not projection_script.exists():
        raise FileNotFoundError(f"feature projection script not found: {projection_script}")

    out_dir = (
        Path(args.feature_projection_out_dir) if args.feature_projection_out_dir else (save_dir / "feature_projection")
    )
    md_path = Path(args.feature_projection_md_path) if args.feature_projection_md_path else (save_dir / "ff_maps.md")
    imgsz = args.feature_projection_imgsz or args.imgsz

    projection_flash_mode = (
        args.flash_mode if args.feature_projection_flash_mode == "same" else args.feature_projection_flash_mode
    )
    env = os.environ.copy()
    _apply_flash_mode_to_env(env, projection_flash_mode)

    cmd = [
        sys.executable,
        str(projection_script),
        "--model",
        str(best_pt),
        "--source",
        str(args.feature_projection_source),
        "--valid-dir",
        str(args.feature_projection_valid_dir),
        "--imgsz",
        str(imgsz),
        "--device",
        str(args.feature_projection_device),
        "--wait-seconds",
        str(args.feature_projection_wait_seconds),
        "--poll-seconds",
        str(args.feature_projection_poll_seconds),
        "--out-dir",
        str(out_dir),
        "--md-path",
        str(md_path),
        "--dataset-name",
        str(args.feature_projection_dataset_name),
        "--model-name",
        str(args.feature_projection_model_name),
        "--variant",
        str(args.feature_projection_variant),
    ]

    print(f"[api-style] feature projection start (post-train). best_pt={best_pt}")
    print(f"[api-style] feature projection flash_mode={projection_flash_mode}")
    if args.feature_projection_log_path:
        log_path = Path(args.feature_projection_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env)
    print(f"[api-style] feature projection complete. out_dir={out_dir} md={md_path}")


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    apply_flash_mode(args.flash_mode)

    from ultralytics import YOLO

    backend = resolve_flash_backend()
    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "task": args.task,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "amp": args.amp,
        "scale": args.scale,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
    }
    if args.time is not None:
        train_kwargs["time"] = args.time
    if args.cache:
        train_kwargs["cache"] = args.cache
    if args.fraction is not None:
        train_kwargs["fraction"] = args.fraction

    train_kwargs = merge_kwarg_sources(train_kwargs, parse_kv_overrides(args.arg), parse_unknown_cli_overrides(unknown))

    print_runtime("train", args.flash_mode, backend, train_kwargs)
    results = model.train(**train_kwargs)
    try:
        from ultralytics.nn.modules import block as block_mod

        if hasattr(block_mod, "format_flash_telemetry_summary"):
            print(block_mod.format_flash_telemetry_summary())
    except Exception as e:
        print(f"[api-style] flash telemetry summary unavailable: {e}")
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    _run_feature_projection(args, save_dir)
    print(f"[api-style] train complete. save_dir={save_dir}")
    print(f"[api-style] train result type={type(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[api-style] train failed: {e}", file=sys.stderr)
        raise
