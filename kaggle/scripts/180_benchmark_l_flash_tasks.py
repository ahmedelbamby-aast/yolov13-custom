#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt

TASKS = {
    "segment": {
        "model": "ultralytics/cfg/models/v13/yolov13l-seg.yaml",
        "data": "coco8-seg.yaml",
        "batch": 16,
        "metric_priority": ["metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/mAP50-95(B)", "metrics/mAP50(B)"],
    },
    "pose": {
        "model": "ultralytics/cfg/models/v13/yolov13l-pose.yaml",
        "data": "coco8-pose.yaml",
        "batch": 16,
        "metric_priority": ["metrics/mAP50-95(P)", "metrics/mAP50(P)", "metrics/mAP50-95(B)", "metrics/mAP50(B)"],
    },
    "obb": {
        "model": "ultralytics/cfg/models/v13/yolov13l-obb.yaml",
        "data": "dota8.yaml",
        "batch": 8,
        "metric_priority": ["metrics/mAP50-95(B)", "metrics/mAP50(B)"],
    },
}

BACKENDS = {
    "fallback": {"use_turing_flash": False, "disable_flash": True},
    "turing": {"use_turing_flash": True, "disable_flash": False},
}


def _ffloat(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_metric(task_name: str, rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = rows[-1].keys()
    for k in TASKS[task_name]["metric_priority"]:
        if k in keys:
            return k
    for k in keys:
        if k.startswith("metrics/"):
            return k
    return ""


def _configure_backend(use_turing_flash: bool, disable_flash: bool):
    os.environ["Y13_USE_TURING_FLASH"] = "1" if use_turing_flash else "0"
    os.environ["Y13_DISABLE_FLASH"] = "1" if disable_flash else "0"

    from ultralytics.nn.modules import block

    if hasattr(block, "configure_flash_backend"):
        block.configure_flash_backend(disable_flash=disable_flash, use_turing_flash=use_turing_flash)

    return block


def run_suite(
    backend_name: str,
    use_turing_flash: bool,
    disable_flash: bool,
    epochs: int,
    imgsz: int,
    workers: int,
    output_root: Path,
) -> dict:
    block = _configure_backend(use_turing_flash=use_turing_flash, disable_flash=disable_flash)

    from ultralytics import YOLO

    suite_root = output_root / backend_name
    suite_root.mkdir(parents=True, exist_ok=True)

    tasks_out = {}
    for task_name, cfg in TASKS.items():
        run_name = f"{task_name}_l_{epochs}e"
        run_start = time.perf_counter()

        YOLO(cfg["model"]).train(
            data=cfg["data"],
            epochs=epochs,
            imgsz=imgsz,
            batch=cfg["batch"],
            workers=workers,
            device="0,1",
            optimizer="AdamW",
            lr0=1e-3,
            weight_decay=5e-4,
            project=str(suite_root),
            name=run_name,
            exist_ok=True,
            amp=True,
        )

        wall_seconds = time.perf_counter() - run_start
        run_dir = suite_root / run_name
        rows = _read_rows(run_dir / "results.csv")
        final = rows[-1] if rows else {}
        metric_key = _pick_metric(task_name, rows)

        item = {
            "task": task_name,
            "model": cfg["model"],
            "data": cfg["data"],
            "batch": cfg["batch"],
            "epochs": epochs,
            "imgsz": imgsz,
            "wall_seconds": wall_seconds,
            "avg_epoch_seconds": wall_seconds / max(1, epochs),
            "results_csv": str(run_dir / "results.csv"),
            "weights": str(run_dir / "weights" / "best.pt"),
            "metric_key": metric_key,
            "metric_value": _ffloat(final.get(metric_key, 0.0)) if metric_key else 0.0,
            "final_metrics": final,
        }
        tasks_out[task_name] = item
        (suite_root / f"{task_name}_metrics.json").write_text(json.dumps(item, indent=2), encoding="utf-8")

    summary = {
        "backend_requested": backend_name,
        "flash_backend": getattr(block, "FLASH_BACKEND", "unknown"),
        "flash_flags": {
            "Y13_USE_TURING_FLASH": os.environ.get("Y13_USE_TURING_FLASH", ""),
            "Y13_DISABLE_FLASH": os.environ.get("Y13_DISABLE_FLASH", ""),
        },
        "tasks": tasks_out,
    }
    (suite_root / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _plot_runtime_grouped(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    fb = [compare["fallback"]["tasks"][t]["wall_seconds"] for t in tasks]
    tu = [compare["turing"]["tasks"][t]["wall_seconds"] for t in tasks]

    x = range(len(tasks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], fb, width=width, label="Fallback", color="#6c757d")
    ax.bar([i + width / 2 for i in x], tu, width=width, label="Turing", color="#2a9d8f")
    ax.set_title("YOLOv13-L Wall Time by Task (Fallback vs Turing)")
    ax.set_ylabel("Wall Time (s)")
    ax.set_xticks(list(x), [t.upper() for t in tasks])
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_wall_time_grouped.png", dpi=220)
    plt.close(fig)


def _plot_speedup(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    speedups = []
    for t in tasks:
        fb = compare["fallback"]["tasks"][t]["wall_seconds"]
        tu = compare["turing"]["tasks"][t]["wall_seconds"]
        speedups.append((fb / tu) if tu > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([t.upper() for t in tasks], speedups, color="#264653")
    ax.axhline(1.0, color="#e76f51", linestyle="--", linewidth=1)
    ax.set_title("YOLOv13-L Speedup by Task (Fallback / Turing)")
    ax.set_ylabel("Speedup (x)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for b, s in zip(bars, speedups):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.3f}x", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_speedup.png", dpi=220)
    plt.close(fig)


def _plot_metric_grouped(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    fb = [compare["fallback"]["tasks"][t]["metric_value"] for t in tasks]
    tu = [compare["turing"]["tasks"][t]["metric_value"] for t in tasks]

    x = range(len(tasks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], fb, width=width, label="Fallback", color="#8d99ae")
    ax.bar([i + width / 2 for i in x], tu, width=width, label="Turing", color="#118ab2")
    ax.set_title("YOLOv13-L Final Primary Metric by Task")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(list(x), [t.upper() for t in tasks])
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_primary_metric_grouped.png", dpi=220)
    plt.close(fig)


def _plot_avg_epoch_grouped(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    fb = [compare["fallback"]["tasks"][t]["avg_epoch_seconds"] for t in tasks]
    tu = [compare["turing"]["tasks"][t]["avg_epoch_seconds"] for t in tasks]

    x = range(len(tasks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], fb, width=width, label="Fallback", color="#adb5bd")
    ax.bar([i + width / 2 for i in x], tu, width=width, label="Turing", color="#06d6a0")
    ax.set_title("YOLOv13-L Avg Epoch Time by Task")
    ax.set_ylabel("Seconds / Epoch")
    ax.set_xticks(list(x), [t.upper() for t in tasks])
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_avg_epoch_grouped.png", dpi=220)
    plt.close(fig)


def _plot_avg_epoch_line(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    x = list(range(len(tasks)))
    fb = [compare["fallback"]["tasks"][t]["avg_epoch_seconds"] for t in tasks]
    tu = [compare["turing"]["tasks"][t]["avg_epoch_seconds"] for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, fb, marker="o", linewidth=2.2, color="#6c757d", label="Fallback")
    ax.plot(x, tu, marker="o", linewidth=2.2, color="#2a9d8f", label="Turing")
    ax.set_title("YOLOv13-L Avg Epoch Time Line Comparison")
    ax.set_ylabel("Seconds / Epoch")
    ax.set_xticks(x, [t.upper() for t in tasks])
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    for xi, y in zip(x, fb):
        ax.text(xi, y + 0.03, f"{y:.2f}s", ha="center", va="bottom", fontsize=9, color="#495057")
    for xi, y in zip(x, tu):
        ax.text(xi, y - 0.05, f"{y:.2f}s", ha="center", va="top", fontsize=9, color="#1d6f67")
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_avg_epoch_line_compare.png", dpi=220)
    plt.close(fig)


def _plot_wall_delta_pct(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    deltas = []
    for task_name in tasks:
        fb = compare["fallback"]["tasks"][task_name]["wall_seconds"]
        tu = compare["turing"]["tasks"][task_name]["wall_seconds"]
        deltas.append(((tu - fb) / fb) * 100.0 if fb > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([t.upper() for t in tasks], deltas, color="#457b9d")
    ax.axhline(0.0, color="#e63946", linestyle="--", linewidth=1)
    ax.set_title("YOLOv13-L Wall-Time Delta (Turing vs Fallback)")
    ax.set_ylabel("Delta % (negative is faster)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for b, d in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width() / 2, d + (0.3 if d >= 0 else -0.8), f"{d:.2f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_wall_delta_pct.png", dpi=220)
    plt.close(fig)


def _plot_curves(compare: dict, out_dir: Path):
    tasks = list(TASKS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)

    for ax, task_name in zip(axes, tasks):
        for backend_name, color in (("fallback", "#6c757d"), ("turing", "#2a9d8f")):
            item = compare[backend_name]["tasks"][task_name]
            rows = _read_rows(Path(item["results_csv"]))
            if not rows:
                continue
            key = item["metric_key"]
            if not key or key not in rows[0]:
                continue
            x = [int(_ffloat(r.get("epoch", i + 1))) for i, r in enumerate(rows)]
            y = [_ffloat(r.get(key, 0.0)) for r in rows]
            ax.plot(x, y, marker="o", linewidth=1.8, label=f"{backend_name} ({key})", color=color)

        ax.set_title(task_name.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Primary Metric")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("YOLOv13-L Metric Curves by Task")
    fig.tight_layout()
    fig.savefig(out_dir / "l_tasks_metric_curves.png", dpi=220)
    plt.close(fig)


def _write_report(compare: dict, out_dir: Path, repo_report: Path):
    total_fb = sum(compare["fallback"]["tasks"][t]["wall_seconds"] for t in TASKS)
    total_tu = sum(compare["turing"]["tasks"][t]["wall_seconds"] for t in TASKS)
    overall_speedup = (total_fb / total_tu) if total_tu > 0 else 0.0

    lines = [
        "# YOLOv13-L Flash Backend Task Comparison",
        "",
        "## Scope",
        "- Model scale: `l` only for all benchmarks.",
        "- Tasks: `segment`, `pose`, `obb`.",
        "- Backends compared: `fallback` vs `flash_attn_turing`.",
        "",
        "## Backend Detection",
        f"- Fallback suite backend: `{compare['fallback']['flash_backend']}`",
        f"- Turing suite backend: `{compare['turing']['flash_backend']}`",
        f"- Total fallback wall time (all tasks): `{total_fb:.2f}s`",
        f"- Total turing wall time (all tasks): `{total_tu:.2f}s`",
        f"- Overall speedup (fallback/turing): `{overall_speedup:.4f}x`",
        "",
        "## Results",
        "",
        "| Task | Batch | Fallback wall (s) | Turing wall (s) | Delta % (tu-fb)/fb | Speedup (fb/tu) | Metric key | Fallback metric | Turing metric |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]

    for task_name in TASKS:
        fb = compare["fallback"]["tasks"][task_name]
        tu = compare["turing"]["tasks"][task_name]
        speedup = (fb["wall_seconds"] / tu["wall_seconds"]) if tu["wall_seconds"] > 0 else 0.0
        delta_pct = (
            ((tu["wall_seconds"] - fb["wall_seconds"]) / fb["wall_seconds"] * 100.0) if fb["wall_seconds"] > 0 else 0.0
        )
        metric_key = tu["metric_key"] or fb["metric_key"] or "n/a"
        lines.append(
            "| {task} | {batch} | {fbw:.2f} | {tuw:.2f} | {dp:.2f}% | {sp:.4f}x | {mk} | {fbm:.4f} | {tum:.4f} |".format(
                task=task_name,
                batch=fb["batch"],
                fbw=fb["wall_seconds"],
                tuw=tu["wall_seconds"],
                dp=delta_pct,
                sp=speedup,
                mk=metric_key,
                fbm=fb["metric_value"],
                tum=tu["metric_value"],
            )
        )

    lines += [
        "",
        "## Visualizations",
        f"- Grouped wall time: `{out_dir / 'l_tasks_wall_time_grouped.png'}`",
        f"- Grouped avg epoch time: `{out_dir / 'l_tasks_avg_epoch_grouped.png'}`",
        f"- Avg epoch line comparison: `{out_dir / 'l_tasks_avg_epoch_line_compare.png'}`",
        f"- Wall-time delta percent: `{out_dir / 'l_tasks_wall_delta_pct.png'}`",
        f"- Speedup bars: `{out_dir / 'l_tasks_speedup.png'}`",
        f"- Final primary metric grouped bars: `{out_dir / 'l_tasks_primary_metric_grouped.png'}`",
        f"- Per-task metric curves: `{out_dir / 'l_tasks_metric_curves.png'}`",
        "",
        "## Artifact Root",
        f"- `{out_dir.parent}`",
    ]

    text = "\n".join(lines) + "\n"
    repo_report.parent.mkdir(parents=True, exist_ok=True)
    repo_report.write_text(text, encoding="utf-8")


def _sync_repo_artifacts(output_root: Path, repo_artifacts_root: Path, epochs: int):
    repo_artifacts_root.parent.mkdir(parents=True, exist_ok=True)
    if repo_artifacts_root.exists():
        shutil.rmtree(repo_artifacts_root)

    (repo_artifacts_root / "plots").mkdir(parents=True, exist_ok=True)
    for p in sorted((output_root / "plots").glob("*.png")):
        shutil.copy2(p, repo_artifacts_root / "plots" / p.name)

    compare_src = output_root / "compare_summary.json"
    if compare_src.exists():
        shutil.copy2(compare_src, repo_artifacts_root / "compare_summary.json")

    for backend_name in BACKENDS:
        src_backend = output_root / backend_name
        dst_backend = repo_artifacts_root / backend_name
        dst_backend.mkdir(parents=True, exist_ok=True)

        for name in ["suite_summary.json", "segment_metrics.json", "pose_metrics.json", "obb_metrics.json"]:
            src = src_backend / name
            if src.exists():
                shutil.copy2(src, dst_backend / name)

        for task_name in TASKS:
            src_csv = src_backend / f"{task_name}_l_{epochs}e" / "results.csv"
            if src_csv.exists():
                shutil.copy2(src_csv, dst_backend / f"{task_name}_results.csv")


def main():
    epochs = int(os.getenv("Y13_BENCH_EPOCHS", "5"))
    imgsz = int(os.getenv("Y13_BENCH_IMGSZ", "640"))
    workers = int(os.getenv("Y13_BENCH_WORKERS", "4"))
    output_root = Path(os.getenv("Y13_BENCH_OUT_ROOT", "/kaggle/working/phase2_l_flash_compare"))
    reuse_existing = os.getenv("Y13_BENCH_REUSE", "0") == "1"
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        fallback = json.loads((output_root / "fallback" / "suite_summary.json").read_text(encoding="utf-8"))
        turing = json.loads((output_root / "turing" / "suite_summary.json").read_text(encoding="utf-8"))
    else:
        fallback = run_suite(
            backend_name="fallback",
            use_turing_flash=BACKENDS["fallback"]["use_turing_flash"],
            disable_flash=BACKENDS["fallback"]["disable_flash"],
            epochs=epochs,
            imgsz=imgsz,
            workers=workers,
            output_root=output_root,
        )
        turing = run_suite(
            backend_name="turing",
            use_turing_flash=BACKENDS["turing"]["use_turing_flash"],
            disable_flash=BACKENDS["turing"]["disable_flash"],
            epochs=epochs,
            imgsz=imgsz,
            workers=workers,
            output_root=output_root,
        )

    compare = {"fallback": fallback, "turing": turing}
    (output_root / "compare_summary.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")

    _plot_runtime_grouped(compare, plots_dir)
    _plot_avg_epoch_grouped(compare, plots_dir)
    _plot_avg_epoch_line(compare, plots_dir)
    _plot_wall_delta_pct(compare, plots_dir)
    _plot_speedup(compare, plots_dir)
    _plot_metric_grouped(compare, plots_dir)
    _plot_curves(compare, plots_dir)

    repo_root = Path(os.getenv("Y13_REPO_ROOT", "/kaggle/work_here/yolov13"))
    repo_artifacts_root = repo_root / "kaggle" / "benchmarks" / "l_flash_tasks"
    repo_report = repo_root / "kaggle" / "reports" / "BENCHMARK_L_FLASH_TASKS_COMPARISON.md"

    _write_report(compare, plots_dir, repo_report)
    _sync_repo_artifacts(output_root, repo_artifacts_root, epochs=epochs)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "repo_artifacts": str(repo_artifacts_root),
                "repo_report": str(repo_report),
                "fallback_backend": fallback["flash_backend"],
                "turing_backend": turing["flash_backend"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
