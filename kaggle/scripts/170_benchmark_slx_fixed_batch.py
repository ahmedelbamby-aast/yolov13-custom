#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

FIXED_BATCHES = {"s": 32, "l": 16, "x": 12}
MODEL_CFG = {
    "s": "ultralytics/cfg/models/v13/yolov13s.yaml",
    "l": "ultralytics/cfg/models/v13/yolov13l.yaml",
    "x": "ultralytics/cfg/models/v13/yolov13x.yaml",
}


def dataset_info(data_yaml: str) -> dict:
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(data_yaml)
    train = d["train"] if not isinstance(d["train"], (list, tuple)) else d["train"][0]
    train_path = Path(train)
    files = [x for x in train_path.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    dims = []
    for f in files[:400]:
        try:
            with Image.open(f) as im:
                dims.append((im.width, im.height))
        except Exception:
            continue

    long_sides = [max(w, h) for w, h in dims]
    short_sides = [min(w, h) for w, h in dims]
    long_mode = Counter(long_sides).most_common(1)[0][0]
    short_mode = Counter(short_sides).most_common(1)[0][0]
    imgsz = max(320, int(round(long_mode / 32) * 32))

    return {
        "dataset": data_yaml,
        "train_dir": str(train_path),
        "samples": len(dims),
        "long_mode": int(long_mode),
        "short_mode": int(short_mode),
        "long_median": int(statistics.median(long_sides)),
        "imgsz": int(imgsz),
        "sample_image": str(files[0]),
    }


def last_metrics(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def ffloat(d: dict, key: str) -> float:
    try:
        return float(d.get(key, 0.0))
    except Exception:
        return 0.0


def save_feature_maps(weights: Path, sample_image: str, imgsz: int, case: str, run_root: Path):
    from ultralytics import YOLO

    fm_root = run_root / "feature_maps"
    fm_root.mkdir(parents=True, exist_ok=True)
    YOLO(str(weights)).predict(
        source=sample_image,
        imgsz=imgsz,
        device=0,
        save=True,
        visualize=True,
        project=str(fm_root),
        name=case,
        exist_ok=True,
    )


def plot_summary(run_root: Path, results: dict, title_suffix: str):
    variants = ["s", "l", "x"]
    wall = [results[v]["wall_seconds"] for v in variants]
    per_epoch = [results[v]["avg_epoch_seconds"] for v in variants]

    p = run_root / "plots"
    p.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([v.upper() for v in variants], wall, color=["#6f42c1", "#8a2be2", "#b566ff"])
    ax.set_title(f"YOLOv13 S/L/X Runtime ({title_suffix})")
    ax.set_ylabel("Wall Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for b, w in zip(bars, wall):
        ax.text(b.get_x() + b.get_width() / 2, w + 1, f"{w:.1f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(p / "slx_fixed_runtime_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([v.upper() for v in variants], per_epoch, marker="o", linewidth=2, color="#8a2be2")
    ax.set_title(f"YOLOv13 S/L/X Avg Seconds per Epoch ({title_suffix})")
    ax.set_ylabel("Seconds / epoch")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(per_epoch):
        ax.text(i, v + 0.03, f"{v:.2f}s", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(p / "slx_fixed_avg_epoch_line.png", dpi=220)
    plt.close(fig)


def run(data_yaml: str = "coco128.yaml", epochs: int = 30, workers: int = 4, use_turing_flash: bool = True, force_disable_flash: bool = False, run_name: str = "y13_bench_slx_30e_turing_fixed") -> dict:
    os.environ["Y13_USE_TURING_FLASH"] = "1" if use_turing_flash else "0"
    os.environ["Y13_DISABLE_FLASH"] = "1" if force_disable_flash else "0"

    from ultralytics import YOLO
    from ultralytics.nn.modules import block

    run_root = Path("/kaggle/working") / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    ds = dataset_info(data_yaml)
    imgsz = ds["imgsz"]

    results = {}
    for v in ["s", "l", "x"]:
        case = f"{v}_bench_fixed"
        start = time.perf_counter()
        YOLO(MODEL_CFG[v]).train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=FIXED_BATCHES[v],
            workers=workers,
            device="0,1",
            optimizer="AdamW",
            lr0=1e-3,
            weight_decay=5e-4,
            project=str(run_root),
            name=case,
            exist_ok=True,
            amp=True,
        )
        elapsed = time.perf_counter() - start

        run_dir = run_root / case
        m = last_metrics(run_dir / "results.csv")
        w = run_dir / "weights" / "best.pt"
        save_feature_maps(w, ds["sample_image"], imgsz, case, run_root)

        item = {
            "variant": v,
            "batch": FIXED_BATCHES[v],
            "imgsz": imgsz,
            "epochs": epochs,
            "wall_seconds": elapsed,
            "avg_epoch_seconds": elapsed / max(1, epochs),
            "mAP50": ffloat(m, "metrics/mAP50(B)"),
            "mAP50_95": ffloat(m, "metrics/mAP50-95(B)"),
            "results_csv": str(run_dir / "results.csv"),
            "weights": str(w),
            "feature_map_dir": str(run_root / "feature_maps" / case),
            "final_metrics": m,
        }
        results[v] = item
        (run_root / f"{v}_metrics.json").write_text(json.dumps(item, indent=2), encoding="utf-8")

    backend = getattr(block, "FLASH_BACKEND", "unknown")
    title_suffix = "Turing Flash fixed batches" if use_turing_flash and not force_disable_flash else "Fallback fixed batches"
    plot_summary(run_root, results, title_suffix)

    summary = {
        "dataset": ds,
        "flash_backend": backend,
        "flash_flags": {
            "Y13_USE_TURING_FLASH": os.environ.get("Y13_USE_TURING_FLASH"),
            "Y13_DISABLE_FLASH": os.environ.get("Y13_DISABLE_FLASH"),
        },
        "fixed_batches": FIXED_BATCHES,
        "variants": results,
    }
    (run_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    out = run(
        data_yaml=os.getenv("Y13_BENCH_DATA", "coco128.yaml"),
        epochs=int(os.getenv("Y13_BENCH_EPOCHS", "30")),
        workers=int(os.getenv("Y13_BENCH_WORKERS", "4")),
        use_turing_flash=os.getenv("Y13_BENCH_USE_TURING_FLASH", "1") == "1",
        force_disable_flash=os.getenv("Y13_BENCH_FORCE_DISABLE_FLASH", "0") == "1",
        run_name=os.getenv("Y13_BENCH_RUN_NAME", "y13_bench_slx_30e_turing_fixed"),
    )
    print(json.dumps({"variants": list(out["variants"].keys()), "backend": out["flash_backend"]}, indent=2))
