#!/usr/bin/env python3
"""Train YOLOv13 models with a simple developer-facing interface."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument(
        "--feature-projection",
        action="store_true",
        help="Run per-layer feature-map projection after training completes using best.pt.",
    )
    parser.add_argument(
        "--feature-projection-script",
        default="kaggle/scripts/37_feature_map_projection.py",
        help="Path to feature projection script.",
    )
    parser.add_argument(
        "--feature-projection-source",
        default="",
        help="Optional single source image for projection. If empty, script auto-picks from valid-dir.",
    )
    parser.add_argument(
        "--feature-projection-valid-dir",
        default="/kaggle/work_here/datasets/roboflow_custom_detect/valid/images",
        help="Fallback valid images directory for source auto-pick.",
    )
    parser.add_argument("--feature-projection-device", default="0", help="Device for projection inference.")
    parser.add_argument(
        "--feature-projection-imgsz",
        type=int,
        default=None,
        help="Projection image size. Defaults to train --imgsz.",
    )
    parser.add_argument(
        "--feature-projection-out-dir",
        default="",
        help="Projection output dir. Defaults to <save_dir>/feature_projection.",
    )
    parser.add_argument(
        "--feature-projection-md-path",
        default="",
        help="Markdown report path. Defaults to <save_dir>/ff_maps.md.",
    )
    parser.add_argument(
        "--feature-projection-log-path",
        default="",
        help="Optional log file for projection stdout/stderr. Defaults to train stdout.",
    )
    parser.add_argument(
        "--feature-projection-dataset-name",
        default="SBAS-AASTMT-AI-ALAMEIN",
        help="Dataset title to stamp on overlays/report.",
    )
    parser.add_argument("--feature-projection-model-name", default="YOLOv13", help="Model title for overlays/report.")
    parser.add_argument("--feature-projection-variant", default="l", help="Model variant label for overlays/report.")
    parser.add_argument(
        "--feature-projection-flash-mode",
        default="same",
        choices=("same", "auto", "fallback", "turing"),
        help="Flash mode for projection phase. 'same' reuses --flash-mode from training.",
    )
    parser.add_argument(
        "--feature-projection-wait-seconds",
        type=int,
        default=300,
        help="Max wait seconds for best.pt in projection script.",
    )
    parser.add_argument(
        "--feature-projection-poll-seconds",
        type=int,
        default=10,
        help="Polling interval seconds while waiting for best.pt.",
    )
    add_flash_mode_arg(parser)
    add_extra_overrides_arg(parser)
    return parser


def _apply_flash_mode_to_env(env: dict[str, str], mode: str) -> None:
    if mode == "fallback":
        env["Y13_DISABLE_FLASH"] = "1"
        env["Y13_USE_TURING_FLASH"] = "0"
    elif mode == "turing":
        env["Y13_DISABLE_FLASH"] = "0"
        env["Y13_USE_TURING_FLASH"] = "1"
    else:
        env["Y13_DISABLE_FLASH"] = "0"
        env.setdefault("Y13_USE_TURING_FLASH", "0")


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

    print(f"[y13] feature projection start (post-train). best_pt={best_pt}")
    print(f"[y13] feature projection flash_mode={projection_flash_mode}")
    if args.feature_projection_log_path:
        log_path = Path(args.feature_projection_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env)
    print(f"[y13] feature projection complete. out_dir={out_dir} md={md_path}")


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
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

    extra_overrides = parse_extra_overrides(args.arg)
    unknown_overrides = parse_unknown_cli_overrides(unknown)
    kwargs = merge_kwarg_sources(kwargs, extra_overrides, unknown_overrides)
    print_runtime_header("train", flash_backend, kwargs)

    model.train(**kwargs)
    try:
        from ultralytics.nn.modules import block as block_mod

        if hasattr(block_mod, "format_flash_telemetry_summary"):
            print(block_mod.format_flash_telemetry_summary())
    except Exception as e:
        print(f"[y13] flash telemetry summary unavailable: {e}")
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    _run_feature_projection(args, save_dir)
    print(f"[y13] training complete. save_dir={save_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[y13] train failed: {e}", file=sys.stderr)
        raise
