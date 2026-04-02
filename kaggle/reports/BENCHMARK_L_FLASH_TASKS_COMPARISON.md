# YOLOv13-L Flash Backend Task Comparison

## Scope
- Model scale: `l` only for all benchmarks.
- Tasks: `detect`, `segment`, `pose`, `obb`.
- Backends compared: `fallback` vs `flash_attn_turing`.
- `imgsz` is task-specific.

## Backend Detection
- Fallback suite backend: `fallback`
- Turing suite backend: `flash_attn_turing`
- Total fallback wall time (all tasks): `622.96s`
- Total turing wall time (all tasks): `613.43s`
- Overall speedup (fallback/turing): `1.0155x`

## Results

| Task | Batch | ImgSz | Fallback wall (s) | Turing wall (s) | Delta % (tu-fb)/fb | Speedup (fb/tu) | Metric key | Fallback metric (peak) | Turing metric (peak) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| detect | 16 | 640 | 148.60 | 144.42 | -2.82% | 1.0290x | metrics/mAP50-95(B) | 0.0000 | 0.0000 |
| segment | 16 | 640 | 150.26 | 149.94 | -0.22% | 1.0022x | metrics/mAP50(M) | 0.0000 | 0.0000 |
| pose | 16 | 640 | 139.36 | 139.76 | 0.29% | 0.9971x | metrics/mAP50(B) | 0.0287 | 0.0287 |
| obb | 4 | 1024 | 184.73 | 179.31 | -2.93% | 1.0302x | metrics/mAP50(B) | 0.0000 | 0.0000 |

## Visualizations
- Grouped wall time: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_wall_time_grouped.png`
- Grouped avg epoch time: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_avg_epoch_grouped.png`
- Avg epoch line comparison: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_avg_epoch_line_compare.png`
- Wall-time delta percent: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_wall_delta_pct.png`
- Speedup bars: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_speedup.png`
- Final primary metric grouped bars: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_primary_metric_grouped.png`
- Per-task metric curves: `/kaggle/working/phase2_l_flash_compare_v2/plots/l_tasks_metric_curves.png`

## Artifact Root
- `/kaggle/working/phase2_l_flash_compare_v2`
