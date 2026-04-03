#!/usr/bin/env python3
"""GPU-first benchmark runner with deterministic format control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _common import (
    add_extra_overrides_arg,
    add_flash_mode_arg,
    apply_flash_mode,
    load_yolo_and_flash_backend,
    merge_kwarg_sources,
    parse_extra_overrides,
    parse_unknown_cli_overrides,
    print_runtime_header,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark YOLOv13 on T4 GPU with controlled export formats.")
    p.add_argument("--model", required=True, help="Model cfg/checkpoint path.")
    p.add_argument("--data", default="coco8.yaml", help="Dataset yaml used for val step inside benchmark.")
    p.add_argument("--imgsz", type=int, default=640, help="Image size.")
    p.add_argument("--device", default="0", help="Benchmark device, keep GPU (e.g. 0).")
    p.add_argument("--half", action="store_true", help="Use FP16 where supported.")
    p.add_argument("--int8", action="store_true", help="Use INT8 where supported.")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Pass benchmark verbose mode (raises on assertion-style failures).",
    )
    p.add_argument(
        "--format",
        action="append",
        default=["onnx", "engine"],
        help="Benchmark formats to run via export+predict+val (repeatable). Defaults: onnx, engine.",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write JSON benchmark summary.",
    )
    add_flash_mode_arg(p)
    add_extra_overrides_arg(p)
    return p


def _run_single_format(model_path: str, task: str, fmt: str, kwargs: dict) -> dict:
    from ultralytics import YOLO

    m = YOLO(model_path, task=task)
    use_half = bool(kwargs.get("half", False) and fmt in {"engine", "torchscript"})

    export_kwargs = {
        "format": fmt,
        "imgsz": kwargs["imgsz"],
        "device": kwargs["device"],
        "half": use_half,
        "int8": kwargs.get("int8", False),
        "dynamic": True,
        "verbose": False,
    }
    export_kwargs.update({k: v for k, v in kwargs.items() if k in {"workspace", "opset", "simplify", "nms"}})
    artifact = m.export(**export_kwargs)

    em = YOLO(artifact, task=task)
    em.predict(
        source="ultralytics/assets/bus.jpg",
        imgsz=kwargs["imgsz"],
        device=kwargs["device"],
        half=use_half,
    )
    res = em.val(
        data=kwargs["data"],
        imgsz=kwargs["imgsz"],
        batch=1,
        device=kwargs["device"],
        half=use_half,
        int8=kwargs.get("int8", False),
        plots=False,
        verbose=False,
    )
    metric_key = {
        "detect": "metrics/mAP50-95(B)",
        "segment": "metrics/mAP50-95(M)",
        "pose": "metrics/mAP50-95(P)",
        "obb": "metrics/mAP50-95(B)",
    }.get(task, "metrics/mAP50-95(B)")
    rd = getattr(res, "results_dict", {}) or {}
    return {
        "format": fmt,
        "artifact": str(artifact),
        "half_used": use_half,
        "metric_key": metric_key,
        "metric": float(rd.get(metric_key, 0.0)),
        "inference_ms": float((getattr(res, "speed", {}) or {}).get("inference", 0.0)),
    }


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
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
    kwargs = merge_kwarg_sources(kwargs, parse_extra_overrides(args.arg), parse_unknown_cli_overrides(unknown))
    print_runtime_header("benchmark", flash_backend, kwargs)

    task = getattr(model, "task", "detect")
    results = []
    for fmt in list(dict.fromkeys(args.format)):
        try:
            item = _run_single_format(args.model, task, fmt, kwargs)
            item["status"] = "ok"
        except Exception as e:
            item = {"format": fmt, "status": "fail", "error": str(e)[:1200]}
        results.append(item)

    df = pd.DataFrame(results)
    print(df)

    summary = {
        "model": args.model,
        "task": task,
        "flash_backend": flash_backend,
        "device": args.device,
        "results": results,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[y13] wrote benchmark summary to {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[y13] benchmark failed: {e}", file=sys.stderr)
        raise
