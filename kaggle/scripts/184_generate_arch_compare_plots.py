#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = [
    "baseline_flash_off",
    "baseline_flash_on",
    "v2_flash_off",
    "v2_flash_on",
]


def _ffloat(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _read_results_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    if not csv_path.exists():
        return [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({(k.strip() if isinstance(k, str) else k): v for k, v in row.items()})
        keys = list(rows[-1].keys()) if rows else []
    return rows, keys


def _pick_metric_key(keys: list[str]) -> str:
    priority = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ]
    for k in priority:
        if k in keys:
            return k
    for k in keys:
        if k.startswith("metrics/"):
            return k
    return ""


def _parse_telemetry(log_text: str) -> dict:
    telemetry_re = re.compile(
        r"flash_telemetry total=(?P<total>\d+) hits=(?P<hits>\d+) fallbacks=(?P<fallbacks>\d+) "
        r"hit_rate=(?P<hit_rate>[0-9.]+)% "
        r"(?:cuda_total=(?P<cuda_total>\d+) cuda_hits=(?P<cuda_hits>\d+) "
        r"cuda_fallbacks=(?P<cuda_fallbacks>\d+) cuda_hit_rate=(?P<cuda_hit_rate>[0-9.]+)% )?"
        r"head_dims=(?P<head_dims>\{.*?\}) "
        r"fallback_reasons=(?P<fallback_reasons>\{.*\})"
    )
    match = None
    for m in telemetry_re.finditer(log_text):
        match = m
    if match is None:
        return {
            "total": 0,
            "hits": 0,
            "fallbacks": 0,
            "hit_rate": 0.0,
            "cuda_total": 0,
            "cuda_hits": 0,
            "cuda_fallbacks": 0,
            "cuda_hit_rate": 0.0,
            "head_dims": {},
            "fallback_reasons": {},
        }
    return {
        "total": int(match.group("total")),
        "hits": int(match.group("hits")),
        "fallbacks": int(match.group("fallbacks")),
        "hit_rate": _ffloat(match.group("hit_rate")),
        "cuda_total": int(match.group("cuda_total") or 0),
        "cuda_hits": int(match.group("cuda_hits") or 0),
        "cuda_fallbacks": int(match.group("cuda_fallbacks") or 0),
        "cuda_hit_rate": _ffloat(match.group("cuda_hit_rate") or 0),
        "head_dims": ast.literal_eval(match.group("head_dims")),
        "fallback_reasons": ast.literal_eval(match.group("fallback_reasons")),
    }


def _read_durations_csv(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) != 3:
                continue
            name, rc, duration = parts
            out[name] = {"rc": int(_ffloat(rc, 1)), "duration_s": int(_ffloat(duration, 0))}
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    durations = _read_durations_csv(out_root / "run_durations.csv")
    summary = {}

    for name in RUNS:
        run_dir = out_root / name
        log_path = out_root / f"{name}.log"
        csv_path = run_dir / "results.csv"

        rows, keys = _read_results_csv(csv_path)
        metric_key = _pick_metric_key(keys)
        metric_peak = max((_ffloat(r.get(metric_key, 0.0)) for r in rows), default=0.0) if metric_key else 0.0
        metric_final = _ffloat(rows[-1].get(metric_key, 0.0)) if (rows and metric_key) else 0.0

        log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        telemetry = _parse_telemetry(log_text)

        summary[name] = {
            "results_csv": str(csv_path),
            "log_path": str(log_path),
            "metric_key": metric_key,
            "metric_peak": metric_peak,
            "metric_final": metric_final,
            "duration_s": durations.get(name, {}).get("duration_s", 0),
            "rc": durations.get(name, {}).get("rc", 1),
            "telemetry": telemetry,
        }

    labels = RUNS
    durations_s = [summary[n]["duration_s"] for n in labels]
    metric_peaks = [summary[n]["metric_peak"] for n in labels]
    hit_rates = [summary[n]["telemetry"]["hit_rate"] for n in labels]
    cuda_hit_rates = [summary[n]["telemetry"]["cuda_hit_rate"] for n in labels]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0][0].bar(labels, durations_s, color=["#6c757d", "#2a9d8f", "#495057", "#1b7f6b"])
    axes[0][0].set_title("Train Duration by Run (s)")
    axes[0][0].grid(axis="y", alpha=0.25)
    axes[0][0].tick_params(axis="x", rotation=20)

    axes[0][1].bar(labels, metric_peaks, color=["#8d99ae", "#118ab2", "#90be6d", "#43aa8b"])
    axes[0][1].set_title("Primary Metric Peak by Run")
    axes[0][1].grid(axis="y", alpha=0.25)
    axes[0][1].tick_params(axis="x", rotation=20)

    axes[1][0].bar(labels, hit_rates, color=["#adb5bd", "#06d6a0", "#ced4da", "#20c997"])
    axes[1][0].set_title("Flash Hit Rate (%)")
    axes[1][0].set_ylim(0, 100)
    axes[1][0].grid(axis="y", alpha=0.25)
    axes[1][0].tick_params(axis="x", rotation=20)

    axes[1][1].bar(labels, cuda_hit_rates, color=["#f4a261", "#e76f51", "#ffbe0b", "#fb8500"])
    axes[1][1].set_title("CUDA Flash Hit Rate (%)")
    axes[1][1].set_ylim(0, 100)
    axes[1][1].grid(axis="y", alpha=0.25)
    axes[1][1].tick_params(axis="x", rotation=20)

    fig.suptitle("Architecture Comparison: baseline vs _2 with flash on/off")
    fig.tight_layout()
    fig.savefig(out_root / "arch_compare_overview.png", dpi=220)
    plt.close(fig)

    report_lines = [
        "# Architecture Comparison Report",
        "",
        "| Run | rc | Duration (s) | Metric key | Metric peak | Flash hit (%) | CUDA hit (%) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]

    for name in RUNS:
        item = summary[name]
        report_lines.append(
            "| {name} | {rc} | {dur} | {mk} | {mp:.4f} | {hr:.2f} | {chr:.2f} |".format(
                name=name,
                rc=item["rc"],
                dur=item["duration_s"],
                mk=item["metric_key"] or "n/a",
                mp=item["metric_peak"],
                hr=item["telemetry"]["hit_rate"],
                chr=item["telemetry"]["cuda_hit_rate"],
            )
        )

    report_lines += [
        "",
        "Artifacts:",
        "",
        "- `arch_compare_overview.png`",
        "- `arch_compare_summary.json`",
        "- `ARCH_COMPARE_REPORT.md`",
    ]

    (out_root / "ARCH_COMPARE_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_root / "arch_compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"out_root": str(out_root), "runs": summary}, indent=2))


if __name__ == "__main__":
    main()
