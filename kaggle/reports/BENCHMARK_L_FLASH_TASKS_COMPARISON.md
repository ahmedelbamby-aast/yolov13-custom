# YOLOv13-L Flash Backend Task Comparison

## Scope
- Model scale: `l` only for all benchmarks.
- Tasks: `detect`, `segment`, `pose`, `obb`.
- Backends compared: `fallback` vs `flash_attn_turing`.
- `imgsz` is task-specific.

## Backend Detection
- Fallback suite backend: `fallback`
- Turing suite backend: `flash_attn_turing`
- Total fallback wall time (all tasks): `287.02s`
- Total turing wall time (all tasks): `278.23s`
- Overall speedup (fallback/turing): `1.0316x`

## Results

| Task | Batch | ImgSz | Fallback wall (s) | Turing wall (s) | Delta % (tu-fb)/fb | Speedup (fb/tu) | Metric key | Fallback metric (peak) | Turing metric (peak) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| detect | 16 | 640 | 73.02 | 64.27 | -11.98% | 1.1361x | metrics/mAP50-95(B) | 0.0000 | 0.0000 |
| segment | 16 | 640 | 70.00 | 69.86 | -0.20% | 1.0020x | metrics/mAP50(M) | 0.0000 | 0.0000 |
| pose | 16 | 640 | 69.33 | 69.72 | 0.57% | 0.9943x | metrics/mAP50(B) | 0.0160 | 0.0160 |
| obb | 4 | 1024 | 74.68 | 74.38 | -0.40% | 1.0040x | metrics/mAP50(B) | 0.0000 | 0.0000 |

## Visualizations
- Grouped wall time: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_wall_time_grouped.png`
- Grouped avg epoch time: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_avg_epoch_grouped.png`
- Avg epoch line comparison: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_avg_epoch_line_compare.png`
- Wall-time delta percent: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_wall_delta_pct.png`
- Speedup bars: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_speedup.png`
- Final primary metric grouped bars: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_primary_metric_grouped.png`
- Per-task metric curves: `/kaggle/working/phase3_l_flash_compare_5e/plots/l_tasks_metric_curves.png`

## Artifact Root
- `/kaggle/working/phase3_l_flash_compare_5e`
