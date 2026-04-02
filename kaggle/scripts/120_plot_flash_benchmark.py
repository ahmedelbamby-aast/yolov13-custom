#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

run_root = Path('/kaggle/working/y13_bench_30e_640')
fallback = json.loads((run_root / 'flash_fallback_metrics.json').read_text(encoding='utf-8'))
turing = json.loads((run_root / 'flash_turing_metrics.json').read_text(encoding='utf-8'))

fb = float(fallback['wall_seconds'])
tu = float(turing['wall_seconds'])
speedup = fb / tu if tu > 0 else 0.0

epochs = int(fallback.get('epochs', 30))
fb_ep = fb / epochs
tu_ep = tu / epochs

out_dir = Path('/kaggle/working/y13_bench_30e_640/plots')
out_dir.mkdir(parents=True, exist_ok=True)

# Bar chart: total runtime
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Fallback', 'Turing Flash']
vals = [fb, tu]
colors = ['#6c757d', '#8a2be2']
bars = ax.bar(labels, vals, color=colors)
ax.set_title('YOLOv13 DDP Benchmark (30 epochs, 640x640)')
ax.set_ylabel('Wall Time (seconds)')
ax.grid(axis='y', linestyle='--', alpha=0.3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v:.2f}s', ha='center', va='bottom', fontsize=10)
ax.text(0.5, max(vals) * 0.92, f'Speedup: {speedup:.4f}x', ha='center', fontsize=11, color='#6f42c1')
fig.tight_layout()
fig.savefig(out_dir / 'flash_runtime_bar.png', dpi=200)
plt.close(fig)

# Per-epoch line chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(['Fallback', 'Turing Flash'], [fb_ep, tu_ep], marker='o', linewidth=2, color='#8a2be2')
ax.set_title('Average Time per Epoch')
ax.set_ylabel('Seconds / epoch')
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.text(0, fb_ep + 0.02, f'{fb_ep:.2f}s', ha='center')
ax.text(1, tu_ep + 0.02, f'{tu_ep:.2f}s', ha='center')
fig.tight_layout()
fig.savefig(out_dir / 'flash_runtime_per_epoch.png', dpi=200)
plt.close(fig)

# Update benchmark report with plot references
report = Path('/kaggle/work_here/yolov13/kaggle/reports/FLASH_BACKEND_BENCHMARK_30E_640.md')
text = report.read_text(encoding='utf-8')
section = """
## Visualizations

- Total runtime bar chart: `/kaggle/working/y13_bench_30e_640/plots/flash_runtime_bar.png`
- Per-epoch runtime plot: `/kaggle/working/y13_bench_30e_640/plots/flash_runtime_per_epoch.png`
"""
if '## Visualizations' not in text:
    text += '\n' + section
report.write_text(text, encoding='utf-8')

print('Saved plots to', out_dir)
