#!/usr/bin/env python3
"""Project per-layer feature maps onto a full image and emit ff_maps.md."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature map projection for one image across all eligible layers.")
    p.add_argument("--model", required=True, help="Model weights path (best.pt).")
    p.add_argument("--source", default="", help="Single source image. If empty, pick first image from valid set.")
    p.add_argument("--valid-dir", default="/kaggle/work_here/datasets/roboflow_custom_detect/valid/images")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--wait-seconds", type=int, default=10800, help="Max wait for model to appear.")
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--out-dir", default="/kaggle/working/phase3_custom_time2/feature_projection")
    p.add_argument("--md-path", default="/kaggle/working/ff_maps.md")
    p.add_argument("--dataset-name", default="SBAS-AASTMT-AI-ALAMEIN")
    p.add_argument("--model-name", default="YOLOv13")
    p.add_argument("--variant", default="l")
    return p.parse_args()


def wait_for_file(path: Path, timeout_s: int, poll_s: int) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for model file: {path}")
        time.sleep(poll_s)


def pick_source(valid_dir: Path) -> Path:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(valid_dir.glob(ext))
    if not images:
        raise FileNotFoundError(f"No images found in {valid_dir}")
    return sorted(images)[0]


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def overlay_map(image_bgr: np.ndarray, fmap: np.ndarray, line1: str, line2: str, line3: str) -> np.ndarray:
    fmap = np.nan_to_num(fmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(fmap.min()), float(fmap.max())
    if mx - mn < 1e-12:
        norm = np.zeros_like(fmap, dtype=np.float32)
    else:
        norm = (fmap - mn) / (mx - mn)

    h, w = image_bgr.shape[:2]
    hm = (norm * 255).astype(np.uint8)
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    out = cv2.addWeighted(image_bgr, 0.60, hm, 0.40, 0.0)

    cv2.putText(out, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line3, (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    wait_for_file(model_path, args.wait_seconds, args.poll_seconds)

    source = Path(args.source) if args.source else pick_source(Path(args.valid_dir))
    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(model_path))
    net = yolo.model.model

    captured: dict[str, np.ndarray] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) and len(out) else out
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.shape[0] >= 1:
                captured[layer_name] = tensor[0].detach().float().mean(0).cpu().numpy()

        return hook

    for layer_name, module in net.named_modules():
        if layer_name:
            handles.append(module.register_forward_hook(make_hook(layer_name)))

    yolo.predict(source=str(source), imgsz=args.imgsz, device=args.device, save=False, verbose=False)

    for h in handles:
        h.remove()

    image_bgr = cv2.imread(str(source))
    if image_bgr is None:
        raise RuntimeError(f"Failed to load source image: {source}")

    rows = []
    line1 = f"Model: {args.model_name} | Variant: {args.variant} | Dataset: {args.dataset_name}"

    for i, (layer_name, fmap) in enumerate(captured.items(), start=1):
        line2 = f"Layer: {layer_name}"
        line3 = f"Source: {source.name}"
        over = overlay_map(image_bgr, fmap, line1, line2, line3)
        fname = f"{i:04d}_{sanitize(layer_name)}.jpg"
        fpath = out_dir / fname
        cv2.imwrite(str(fpath), over)
        rows.append(
            {
                "idx": i,
                "layer": layer_name,
                "shape": list(fmap.shape),
                "image": str(fpath),
                "image_rel": str(fpath).replace("/kaggle/working/", ""),
            }
        )

    meta = {
        "model": str(model_path),
        "source": str(source),
        "total_layers": len(rows),
        "dataset": args.dataset_name,
        "projection": {
            "mode": "full_image_overlay",
            "enlargement_enabled": True,
            "resize_interpolation": "INTER_CUBIC",
        },
        "rows": rows,
    }
    (out_dir / "feature_projection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md = Path(args.md_path)
    lines = [
        "# Feature Map Projection",
        "",
        f"- Model: `{args.model_name}`",
        f"- Variant: `{args.variant}`",
        f"- Dataset: `{args.dataset_name}`",
        f"- Source Image: `{source}`",
        f"- Total Layers Projected: `{len(rows)}`",
        "- Enlargement: `enabled` (feature maps are resized to full-image with `INTER_CUBIC`)",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## Layer {row['idx']}: `{row['layer']}`",
                f"- Feature Shape: `{row['shape']}`",
                f"![{row['layer']}]({row['image_rel']})",
                "",
            ]
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "md": str(md), "layers": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
