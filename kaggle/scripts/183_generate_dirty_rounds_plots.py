#!/usr/bin/env python3
"""Generate high-quality round-specific and comparison plots for dirty smoke benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROUND1_SUMMARY = {
    "baseline": {
        "total": 10152,
        "hits": 0,
        "fallbacks": 10152,
        "hit_rate": 0.0,
        "head_dims": {"32": 10152},
        "fallback_reasons": {"not_cuda": 24, "unsupported_head_dim_32": 10128},
        "inference_ms": 3.0,
        "epochs": 5,
        "train_time_hours": 0.164,
        "train_epoch_s": 118.08,
        "log_path": "/kaggle/working/head32_dirty_smoke/baseline.log",
    },
    "head32_enabled": {
        "total": 10152,
        "hits": 10128,
        "fallbacks": 24,
        "hit_rate": 99.76,
        "head_dims": {"32": 10152},
        "fallback_reasons": {"not_cuda": 24},
        "inference_ms": 2.5,
        "epochs": 5,
        "train_time_hours": 0.161,
        "train_epoch_s": 115.92,
        "log_path": "/kaggle/working/head32_dirty_smoke/head32_enabled.log",
    },
    "delta": {
        "hit_rate_pp": 99.76,
        "inference_ms": -0.5,
        "train_epoch_s": -2.16,
        "train_speedup_x": 1.019,
    },
}


def _fmt(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _plot_round(summary: dict, out_png: Path, title: str) -> None:
    b = summary["baseline"]
    h = summary["head32_enabled"]
    speedup = summary.get("delta", {}).get("train_speedup_x")

    labels = ["baseline", "head32_enabled"]
    hit_vals = [b.get("hit_rate", 0.0), h.get("hit_rate", 0.0)]
    inf_vals = [b.get("inference_ms", 0.0) or 0.0, h.get("inference_ms", 0.0) or 0.0]
    trn_vals = [b.get("train_epoch_s", 0.0) or 0.0, h.get("train_epoch_s", 0.0) or 0.0]

    color = "#1f77b4"
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))

    bars0 = axes[0].bar(labels, hit_vals, color=color)
    axes[0].set_title("Flash Hit Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis="y", alpha=0.25)

    bars1 = axes[1].bar(labels, inf_vals, color=color)
    axes[1].set_title("Validation Inference (ms/img)")
    axes[1].grid(axis="y", alpha=0.25)

    bars2 = axes[2].bar(labels, trn_vals, color=color)
    axes[2].set_title("Train Time Per Epoch (s)")
    axes[2].grid(axis="y", alpha=0.25)

    for ax, bars, nd in ((axes[0], bars0, 2), (axes[1], bars1, 3), (axes[2], bars2, 2)):
        ymax = max((bar.get_height() for bar in bars), default=0.0)
        offset = 0.02 * ymax if ymax > 0 else 0.02
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.{nd}f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    if speedup is not None and trn_vals[0] > 0:
        y_line = trn_vals[0]
        axes[2].axhline(y=y_line, color="orange", linestyle="--", linewidth=2)
        axes[2].text(
            0.5,
            y_line * 1.02,
            f"speedup {speedup:.3f}x",
            color="orange",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


def _write_round_report(summary: dict, out_md: Path, title: str, include_cuda: bool) -> None:
    b = summary["baseline"]
    h = summary["head32_enabled"]
    d = summary.get("delta", {})
    lines = [
        f"# {title}",
        "",
        "| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train epoch (s) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| baseline | {_fmt(b.get('hit_rate'))} | {b.get('hits')} | {b.get('fallbacks')} | {_fmt(b.get('inference_ms'), 3)} | {_fmt(b.get('train_epoch_s'))} |",
        f"| head32 enabled | {_fmt(h.get('hit_rate'))} | {h.get('hits')} | {h.get('fallbacks')} | {_fmt(h.get('inference_ms'), 3)} | {_fmt(h.get('train_epoch_s'))} |",
        "",
        "## Delta (enabled - baseline)",
        "",
        f"- flash hit-rate: {_fmt(d.get('hit_rate_pp'))} pp",
        f"- val inference: {_fmt(d.get('inference_ms'), 3)} ms/img",
        f"- train epoch time: {_fmt(d.get('train_epoch_s'), 3)} s",
        f"- speedup: {_fmt(d.get('train_speedup_x'), 3)}x",
        "",
        "## Fallback reasons",
        "",
        f"- baseline: `{b.get('fallback_reasons')}`",
        f"- head32 enabled: `{h.get('fallback_reasons')}`",
    ]
    if include_cuda:
        lines.extend(
            [
                "",
                "## CUDA-only telemetry",
                "",
                f"- baseline cuda hit-rate: {_fmt(b.get('cuda_hit_rate'))}% ({b.get('cuda_hits')}/{b.get('cuda_total')})",
                f"- head32 cuda hit-rate: {_fmt(h.get('cuda_hit_rate'))}% ({h.get('cuda_hits')}/{h.get('cuda_total')})",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bench_root = repo_root / "kaggle" / "benchmarks"

    src_round2_summary = bench_root / "flash_head32_dirty_smoke" / "compare_summary.json"
    round2_summary = json.loads(src_round2_summary.read_text(encoding="utf-8"))

    round1_dir = bench_root / "flash_head32_dirty_smoke_round1"
    round2_dir = bench_root / "flash_head32_dirty_smoke_round2"
    cmp_dir = bench_root / "flash_head32_dirty_smoke_comparison"
    for d in (round1_dir, round2_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    (round1_dir / "compare_summary.json").write_text(json.dumps(ROUND1_SUMMARY, indent=2), encoding="utf-8")
    (round2_dir / "compare_summary.json").write_text(json.dumps(round2_summary, indent=2), encoding="utf-8")

    _plot_round(ROUND1_SUMMARY, round1_dir / "telemetry_compare.png", "Dirty Smoke Round 1 (5 epochs, fraction=0.05)")
    _plot_round(
        round2_summary,
        round2_dir / "telemetry_compare.png",
        "Dirty Smoke Round 2 (parallel split-GPU, 5 epochs, fraction=0.05)",
    )
    _write_round_report(ROUND1_SUMMARY, round1_dir / "REPORT.md", "Dirty Smoke Round 1", include_cuda=False)
    _write_round_report(round2_summary, round2_dir / "REPORT.md", "Dirty Smoke Round 2", include_cuda=True)

    comparison = {
        "round1": {
            "hit_rate_pp": ROUND1_SUMMARY["delta"]["hit_rate_pp"],
            "inference_ms_delta": ROUND1_SUMMARY["delta"]["inference_ms"],
            "train_epoch_s_delta": ROUND1_SUMMARY["delta"]["train_epoch_s"],
            "speedup_x": ROUND1_SUMMARY["delta"]["train_speedup_x"],
        },
        "round2": {
            "hit_rate_pp": round2_summary["delta"]["hit_rate_pp"],
            "inference_ms_delta": round2_summary["delta"]["inference_ms"],
            "train_epoch_s_delta": round2_summary["delta"]["train_epoch_s"],
            "speedup_x": round2_summary["delta"]["train_speedup_x"],
        },
    }
    (cmp_dir / "compare_rounds.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 4, figsize=(30, 7.5))
    labels = ["round1", "round2"]
    color = "#1f77b4"
    metrics = [
        ("hit_rate_pp", "Flash Hit-Rate Gain (pp)"),
        ("inference_ms_delta", "Inference Delta (ms/img)"),
        ("train_epoch_s_delta", "Train Epoch Delta (s)"),
        ("speedup_x", "Speedup (x)"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        vals = [comparison[r][key] for r in labels]
        bars = ax.bar(labels, vals, color=color)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ymax = max(abs(v) for v in vals) if vals else 1.0
        offset = 0.03 * (ymax if ymax else 1.0)
        for bar, val in zip(bars, vals):
            y = val + (offset if val >= 0 else -offset)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=12,
            )

    fig.suptitle("Dirty Smoke Benchmark Comparison: Round1 vs Round2", fontsize=17)
    fig.tight_layout()
    fig.savefig(cmp_dir / "rounds_compare.png", dpi=400)
    plt.close(fig)

    cmp_report = [
        "# Dirty Smoke Rounds Comparison",
        "",
        "| Metric | Round 1 | Round 2 |",
        "|---|---:|---:|",
        f"| Flash hit-rate gain (pp) | {comparison['round1']['hit_rate_pp']:.2f} | {comparison['round2']['hit_rate_pp']:.2f} |",
        f"| Inference delta (ms/img) | {comparison['round1']['inference_ms_delta']:.3f} | {comparison['round2']['inference_ms_delta']:.3f} |",
        f"| Train epoch delta (s) | {comparison['round1']['train_epoch_s_delta']:.3f} | {comparison['round2']['train_epoch_s_delta']:.3f} |",
        f"| Speedup (x) | {comparison['round1']['speedup_x']:.3f} | {comparison['round2']['speedup_x']:.3f} |",
        "",
        "Artifacts:",
        "",
        "- `kaggle/benchmarks/flash_head32_dirty_smoke_round1/telemetry_compare.png`",
        "- `kaggle/benchmarks/flash_head32_dirty_smoke_round2/telemetry_compare.png`",
        "- `kaggle/benchmarks/flash_head32_dirty_smoke_comparison/rounds_compare.png`",
    ]
    (cmp_dir / "REPORT.md").write_text("\n".join(cmp_report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
