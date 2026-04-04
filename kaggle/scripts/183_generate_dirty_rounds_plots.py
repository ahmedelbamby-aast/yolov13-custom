#!/usr/bin/env python3
"""Generate high-quality round-specific and comparison plots for dirty smoke benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"


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

    bar_colors = ["#1f77b4", "#ff7f0e"]
    fig, axes = plt.subplots(1, 3, figsize=(26, 8.5))

    bars0 = axes[0].bar(labels, hit_vals, color=bar_colors)
    axes[0].set_title("Flash Hit Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis="y", alpha=0.25)

    bars1 = axes[1].bar(labels, inf_vals, color=bar_colors)
    axes[1].set_title("Validation Inference (ms/img)")
    axes[1].grid(axis="y", alpha=0.25)

    bars2 = axes[2].bar(labels, trn_vals, color=bar_colors)
    axes[2].set_title("Train Time Per Epoch (s)")
    axes[2].grid(axis="y", alpha=0.25)

    for ax, bars, nd in ((axes[0], bars0, 2), (axes[1], bars1, 3), (axes[2], bars2, 2)):
        ymax = max((bar.get_height() for bar in bars), default=0.0)
        offset = 0.02 * ymax if ymax > 0 else 0.02
        for bar in bars:
            val = bar.get_height()
            color = bar.get_facecolor()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.{nd}f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=color,
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

    fig.suptitle(title, fontsize=18)
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

    labels = ["round1", "round2"]
    r1_base = ROUND1_SUMMARY["baseline"]
    r2_base = round2_summary["baseline"]
    r1_h32 = ROUND1_SUMMARY["head32_enabled"]
    r2_h32 = round2_summary["head32_enabled"]

    base_unsupported = [
        r1_base.get("fallback_reasons", {}).get("unsupported_head_dim_32", 0),
        r2_base.get("fallback_reasons", {}).get("unsupported_head_dim_32", 0),
    ]
    base_not_cuda = [
        r1_base.get("fallback_reasons", {}).get("not_cuda", 0),
        r2_base.get("fallback_reasons", {}).get("not_cuda", 0),
    ]
    h32_flash = [r1_h32.get("hits", 0), r2_h32.get("hits", 0)]
    h32_not_cuda = [
        r1_h32.get("fallback_reasons", {}).get("not_cuda", 0),
        r2_h32.get("fallback_reasons", {}).get("not_cuda", 0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(30, 8.5))

    c_unsupported = "#d62728"
    c_not_cuda = "#9467bd"
    c_flash = "#2ca02c"

    bars_base_uns = axes[0].bar(labels, base_unsupported, color=c_unsupported, label="unsupported_head_dim_32")
    bars_base_nc = axes[0].bar(labels, base_not_cuda, bottom=base_unsupported, color=c_not_cuda, label="not_cuda")
    axes[0].set_title("Baseline Attention Outcome (Stacked)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    bars_h32_flash = axes[1].bar(labels, h32_flash, color=c_flash, label="flash_hits")
    bars_h32_nc = axes[1].bar(labels, h32_not_cuda, bottom=h32_flash, color=c_not_cuda, label="not_cuda")
    axes[1].set_title("Head32 Attention Outcome (Stacked)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    def annotate_stacked(ax, lower_bars, upper_bars):
        for bar in lower_bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(1.0, 0.005 * val),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=bar.get_facecolor(),
            )
        for low, up in zip(lower_bars, upper_bars):
            val = up.get_height()
            top = low.get_height() + val
            ax.text(
                up.get_x() + up.get_width() / 2,
                top + max(1.0, 0.005 * top),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=up.get_facecolor(),
            )

    annotate_stacked(axes[0], bars_base_uns, bars_base_nc)
    annotate_stacked(axes[1], bars_h32_flash, bars_h32_nc)

    fig.suptitle("Dirty Smoke Benchmark Comparison (Stacked): Round1 vs Round2", fontsize=18)
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
