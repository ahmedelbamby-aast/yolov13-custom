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


def _dataset_info(data_yaml: str):
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(data_yaml)
    train = d["train"] if not isinstance(d["train"], (list, tuple)) else d["train"][0]
    train_path = Path(train)
    files = [x for x in train_path.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not files:
        raise RuntimeError(f"No images found in train path: {train_path}")

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
    long_median = int(statistics.median(long_sides))
    imgsz = max(320, int(round(long_mode / 32) * 32))

    return {
        "dataset": data_yaml,
        "train_dir": str(train_path),
        "samples": len(dims),
        "long_mode": int(long_mode),
        "short_mode": int(short_mode),
        "long_median": int(long_median),
        "imgsz": int(imgsz),
        "sample_image": str(files[0]),
    }


def _train_probe(model_cfg: str, data_yaml: str, imgsz: int, batch: int, workers: int, out_root: Path, name: str) -> bool:
    from ultralytics import YOLO

    try:
        YOLO(model_cfg).train(
            data=data_yaml,
            epochs=1,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device="0,1",
            project=str(out_root),
            name=name,
            exist_ok=True,
            amp=True,
            val=False,
            save=False,
            plots=False,
            fraction=0.2,
            verbose=False,
        )
        return True
    except Exception as e:
        m = str(e).lower()
        if "out of memory" in m or "cuda error" in m:
            return False
        raise


def _best_batch(variant: str, model_cfg: str, data_yaml: str, imgsz: int, workers: int, tune_root: Path) -> int:
    candidates = {
        "s": [8, 12, 16, 20, 24, 28, 32],
        "l": [4, 6, 8, 10, 12, 14, 16],
        "x": [2, 4, 6, 8, 10, 12],
    }[variant]
    best = None
    for b in candidates:
        ok = _train_probe(model_cfg, data_yaml, imgsz, b, workers, tune_root, f"tune_{variant}_b{b}")
        if ok:
            best = b
        else:
            break
    if best is None:
        raise RuntimeError(f"No working batch for {variant}")
    return int(best)


def _last_metrics(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _to_float(d: dict, key: str) -> float:
    try:
        return float(d.get(key, 0.0))
    except Exception:
        return 0.0

def _feature_maps(weights: Path, sample_image: str, imgsz: int, case: str, out_root: Path):
    from ultralytics import YOLO

    fm_root = out_root / "feature_maps"
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


def _plots(run_root: Path, data: dict):
    variants = ["s", "l", "x"]
    wall = [data[v]["wall_seconds"] for v in variants]
    batch = [data[v]["batch"] for v in variants]
    map50 = [data[v]["mAP50"] for v in variants]

    p = run_root / "plots"
    p.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([v.upper() for v in variants], wall, color=["#4e79a7", "#f28e2b", "#8a2be2"])
    ax.set_title("YOLOv13 S/L/X Runtime (30 epochs, DDP 2xT4)")
    ax.set_ylabel("Wall Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for b, w, bs in zip(bars, wall, batch):
        ax.text(b.get_x() + b.get_width() / 2, w + 1, f"{w:.1f}s\nbs={bs}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(p / "slx_runtime_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([v.upper() for v in variants], map50, marker="o", linewidth=2, color="#8a2be2")
    ax.set_title("YOLOv13 S/L/X Final mAP50")
    ax.set_ylabel("mAP50")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i, m in enumerate(map50):
        ax.text(i, m + 0.003, f"{m:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(p / "slx_map50_line.png", dpi=220)
    plt.close(fig)

def run_benchmark(data_yaml: str = "coco128.yaml", epochs: int = 30, workers: int = 4) -> dict:
    from ultralytics import YOLO
    from ultralytics.nn.modules import block

    os.environ.setdefault("Y13_USE_TURING_FLASH", "1")
    os.environ["Y13_DISABLE_FLASH"] = "0"

    run_root = Path("/kaggle/working/y13_bench_slx_30e")
    run_root.mkdir(parents=True, exist_ok=True)
    tune_root = run_root / "tune"
    tune_root.mkdir(parents=True, exist_ok=True)

    ds = _dataset_info(data_yaml)
    imgsz = ds["imgsz"]

    models = {
        "s": "ultralytics/cfg/models/v13/yolov13s.yaml",
        "l": "ultralytics/cfg/models/v13/yolov13l.yaml",
        "x": "ultralytics/cfg/models/v13/yolov13x.yaml",
    }

    results = {}
    for v, cfg in models.items():
        b = _best_batch(v, cfg, data_yaml, imgsz, workers, tune_root)
        start = time.perf_counter()
        YOLO(cfg).train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=b,
            workers=workers,
            device="0,1",
            optimizer="AdamW",
            lr0=1e-3,
            weight_decay=5e-4,
            project=str(run_root),
            name=f"{v}_bench",
            exist_ok=True,
            amp=True,
        )
        elapsed = time.perf_counter() - start

        run_dir = run_root / f"{v}_bench"
        m = _last_metrics(run_dir / "results.csv")
        weights = run_dir / "weights" / "best.pt"
        _feature_maps(weights, ds["sample_image"], imgsz, f"{v}_bench", run_root)

        item = {
            "variant": v,
            "batch": b,
            "imgsz": imgsz,
            "epochs": epochs,
            "wall_seconds": elapsed,
            "mAP50": _to_float(m, "metrics/mAP50(B)"),
            "mAP50_95": _to_float(m, "metrics/mAP50-95(B)"),
            "results_csv": str(run_dir / "results.csv"),
            "weights": str(weights),
            "feature_map_dir": str(run_root / "feature_maps" / f"{v}_bench"),
            "final_metrics": m,
        }
        results[v] = item
        (run_root / f"{v}_metrics.json").write_text(json.dumps(item, indent=2), encoding="utf-8")

    _plots(run_root, results)

    summary = {
        "dataset": ds,
        "flash_backend": getattr(block, "FLASH_BACKEND", "unknown"),
        "variants": results,
    }
    (run_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = Path("/kaggle/work_here/yolov13/kaggle/reports/BENCHMARK_SLX_30E.md")
    lines = [
        "# YOLOv13 S/L/X Benchmark (DDP 2xT4)",
        "",
        "## Dataset / imgsz alignment",
        f"- Dataset: `{data_yaml}`",
        f"- Samples checked: `{ds['samples']}`",
        f"- Long side mode: `{ds['long_mode']}`",
        f"- Short side mode: `{ds['short_mode']}`",
        f"- Selected imgsz: `{imgsz}`",
        "",
        "## Flash backend",
        f"- `{summary['flash_backend']}`",
        "",
        "## Results",
    ]
    for v in ("s", "l", "x"):
        r = results[v]
        lines.append(
            f"- {v.upper()}: batch `{r['batch']}`, wall `{r['wall_seconds']:.2f}s`, mAP50 `{r['mAP50']:.4f}`, mAP50-95 `{r['mAP50_95']:.4f}`"
        )
    lines += [
        "",
        "## Artifacts",
        "- `/kaggle/working/y13_bench_slx_30e/plots/slx_runtime_bar.png`",
        "- `/kaggle/working/y13_bench_slx_30e/plots/slx_map50_line.png`",
        "- `/kaggle/working/y13_bench_slx_30e/feature_maps`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = run_benchmark(
        data_yaml=os.getenv("Y13_BENCH_DATA", "coco128.yaml"),
        epochs=int(os.getenv("Y13_BENCH_EPOCHS", "30")),
        workers=int(os.getenv("Y13_BENCH_WORKERS", "4")),
    )
    print(json.dumps({"variants": list(out["variants"].keys()), "backend": out["flash_backend"]}, indent=2))
