#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python: ${PY}" >&2
  exit 1
fi

DATA_YAML="${Y13_DIRTY_DATA_YAML:-${Y13_WORKDIR}/datasets/roboflow_custom_detect_dirty/data.yaml}"
EPOCHS="${Y13_HEAD32_BENCH_EPOCHS:-5}"
FRACTION="${Y13_HEAD32_BENCH_FRACTION:-0.05}"
BATCH="${Y13_HEAD32_BENCH_BATCH:-16}"
IMGSZ="${Y13_HEAD32_BENCH_IMGSZ:-640}"
WORKERS="${Y13_HEAD32_BENCH_WORKERS:-8}"
DEVICE="${Y13_HEAD32_BENCH_DEVICE:-0}"
OUT_ROOT="${Y13_HEAD32_BENCH_OUT_ROOT:-${Y13_OUTPUT_DIR}/head32_dirty_smoke}"

mkdir -p "${OUT_ROOT}"

BASE_LOG="${OUT_ROOT}/baseline.log"
HEAD32_LOG="${OUT_ROOT}/head32_enabled.log"

echo "[head32_dirty_smoke] data=${DATA_YAML} epochs=${EPOCHS} fraction=${FRACTION}"

Y13_DISABLE_FLASH=0 Y13_USE_TURING_FLASH=1 Y13_ENABLE_TURING_HEAD_DIM32=0 \
"${PY}" "${Y13_ROOT}/scripts/train.py" \
  --model ultralytics/cfg/models/v13/yolov13n.yaml \
  --data "${DATA_YAML}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device "${DEVICE}" \
  --workers "${WORKERS}" \
  --project "${OUT_ROOT}" \
  --name baseline \
  --flash-mode turing \
  --arg fraction="${FRACTION}" \
  --arg plots=false \
  > "${BASE_LOG}" 2>&1

Y13_DISABLE_FLASH=0 Y13_USE_TURING_FLASH=1 Y13_ENABLE_TURING_HEAD_DIM32=1 \
"${PY}" "${Y13_ROOT}/scripts/train.py" \
  --model ultralytics/cfg/models/v13/yolov13n.yaml \
  --data "${DATA_YAML}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device "${DEVICE}" \
  --workers "${WORKERS}" \
  --project "${OUT_ROOT}" \
  --name head32_enabled \
  --flash-mode turing \
  --arg fraction="${FRACTION}" \
  --arg plots=false \
  > "${HEAD32_LOG}" 2>&1

export Y13_BENCH_OUT_ROOT="${OUT_ROOT}"
export Y13_BENCH_ARTIFACT_DIR="${Y13_ROOT}/kaggle/benchmarks/flash_head32_dirty_smoke"
export Y13_BENCH_FRACTION="${FRACTION}"

"${PY}" - <<'PY'
import ast
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

out_root = Path(os.environ["Y13_BENCH_OUT_ROOT"])
artifact_dir = Path(os.environ["Y13_BENCH_ARTIFACT_DIR"])
fraction = os.environ["Y13_BENCH_FRACTION"]

logs = {
    "baseline": out_root / "baseline.log",
    "head32_enabled": out_root / "head32_enabled.log",
}
artifact_dir.mkdir(parents=True, exist_ok=True)

telemetry_re = re.compile(
    r"flash_telemetry total=(?P<total>\d+) hits=(?P<hits>\d+) fallbacks=(?P<fallbacks>\d+) "
    r"hit_rate=(?P<hit_rate>[0-9.]+)% head_dims=(?P<head_dims>\{.*?\}) "
    r"fallback_reasons=(?P<fallback_reasons>\{.*\})"
)
speed_re = re.compile(r"Speed:\s+[^\n]*?,\s*(?P<inference>[0-9.]+)ms inference,")
epoch_re = re.compile(r"(\d+)\s+epochs completed in\s+([0-9.]+) hours\.")

summary = {}
for name, path in logs.items():
    txt = path.read_text(encoding="utf-8", errors="ignore")

    tm = None
    for m in telemetry_re.finditer(txt):
        tm = m
    if tm is None:
        raise RuntimeError(f"telemetry summary not found: {path}")

    sm = None
    for m in speed_re.finditer(txt):
        sm = m

    em = None
    for m in epoch_re.finditer(txt):
        em = m

    epoch_count = int(em.group(1)) if em else None
    hours = float(em.group(2)) if em else None
    train_epoch_s = (hours * 3600.0 / epoch_count) if (hours is not None and epoch_count and epoch_count > 0) else None

    summary[name] = {
        "total": int(tm.group("total")),
        "hits": int(tm.group("hits")),
        "fallbacks": int(tm.group("fallbacks")),
        "hit_rate": float(tm.group("hit_rate")),
        "head_dims": ast.literal_eval(tm.group("head_dims")),
        "fallback_reasons": ast.literal_eval(tm.group("fallback_reasons")),
        "inference_ms": float(sm.group("inference")) if sm else None,
        "epochs": epoch_count,
        "train_time_hours": hours,
        "train_epoch_s": train_epoch_s,
        "log_path": str(path),
    }

b = summary["baseline"]
h = summary["head32_enabled"]

speedup = None
if b["train_epoch_s"] and h["train_epoch_s"] and h["train_epoch_s"] > 0:
    speedup = b["train_epoch_s"] / h["train_epoch_s"]

delta = {
    "hit_rate_pp": round(h["hit_rate"] - b["hit_rate"], 2),
    "inference_ms": None if (b["inference_ms"] is None or h["inference_ms"] is None) else round(h["inference_ms"] - b["inference_ms"], 3),
    "train_epoch_s": None if (b["train_epoch_s"] is None or h["train_epoch_s"] is None) else round(h["train_epoch_s"] - b["train_epoch_s"], 3),
    "train_speedup_x": None if speedup is None else round(speedup, 3),
}

compare = {"baseline": b, "head32_enabled": h, "delta": delta}
(artifact_dir / "compare_summary.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")

labels = ["baseline", "head32_enabled"]
hit_rates = [summary[k]["hit_rate"] for k in labels]
infer_ms = [summary[k]["inference_ms"] or 0.0 for k in labels]
train_epoch_s_vals = [summary[k]["train_epoch_s"] or 0.0 for k in labels]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].bar(labels, hit_rates, color=["#8c564b", "#2ca02c"])
axes[0].set_title("Flash Hit Rate (%)")
axes[0].set_ylim(0, 100)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(labels, infer_ms, color=["#1f77b4", "#ff7f0e"])
axes[1].set_title("Validation Inference (ms/img)")
axes[1].grid(axis="y", alpha=0.25)

bars = axes[2].bar(labels, train_epoch_s_vals, color=["#9467bd", "#17becf"])
axes[2].set_title("Train Time Per Epoch (s)")
axes[2].grid(axis="y", alpha=0.25)

if speedup is not None:
    y_line = train_epoch_s_vals[0]
    axes[2].axhline(y=y_line, color="orange", linestyle="--", linewidth=2)
    axes[2].text(
        0.5,
        y_line * 1.01,
        f"speedup {speedup:.2f}x",
        color="orange",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

for bar in bars:
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("turFlash head_dim=32 dirty-data smoke comparison")
fig.tight_layout()
fig.savefig(artifact_dir / "telemetry_compare.png", dpi=170)
plt.close(fig)

def f2(v):
    return "n/a" if v is None else f"{v:.2f}"

def f3(v):
    return "n/a" if v is None else f"{v:+.3f}"

speedup_text = "n/a" if delta["train_speedup_x"] is None else f"{delta['train_speedup_x']:.2f}x"

lines = [
    "# turFlash head_dim=32 dirty-data smoke comparison",
    "",
    "- baseline: `Y13_ENABLE_TURING_HEAD_DIM32=0`",
    "- enabled: `Y13_ENABLE_TURING_HEAD_DIM32=1`",
    f"- epochs: {b['epochs']}",
    "- dataset: remapped dirty Roboflow subset (`student`, `teacher`)",
    f"- fraction: {fraction}",
    "",
    "## Results",
    "",
    "| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train epoch (s) |",
    "|---|---:|---:|---:|---:|---:|",
    f"| baseline | {f2(b['hit_rate'])} | {b['hits']} | {b['fallbacks']} | {f2(b['inference_ms'])} | {f2(b['train_epoch_s'])} |",
    f"| head32 enabled | {f2(h['hit_rate'])} | {h['hits']} | {h['fallbacks']} | {f2(h['inference_ms'])} | {f2(h['train_epoch_s'])} |",
    "",
    "## Delta (enabled - baseline)",
    "",
    f"- flash hit-rate: {delta['hit_rate_pp']:+.2f} percentage points",
    f"- val inference: {f3(delta['inference_ms'])} ms/img",
    f"- train epoch time: {f3(delta['train_epoch_s'])} s",
    f"- train speedup: {speedup_text}",
    "",
    "## Fallback reasons",
    "",
    f"- baseline: `{b['fallback_reasons']}`",
    f"- head32 enabled: `{h['fallback_reasons']}`",
    "",
    "Artifacts:",
    "",
    "- `kaggle/benchmarks/flash_head32_dirty_smoke/compare_summary.json`",
    "- `kaggle/benchmarks/flash_head32_dirty_smoke/telemetry_compare.png`",
    "- `kaggle/benchmarks/flash_head32_dirty_smoke/REPORT.md`",
]
(artifact_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(compare, indent=2))
print("artifact_dir", artifact_dir)
PY

unset Y13_BENCH_OUT_ROOT Y13_BENCH_ARTIFACT_DIR Y13_BENCH_FRACTION

echo "[head32_dirty_smoke] done"
