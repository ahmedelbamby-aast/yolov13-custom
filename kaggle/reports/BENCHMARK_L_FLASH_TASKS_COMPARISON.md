# YOLOv13-L Flash Backend Task Comparison

## Scope
- Model scale: `l` only for all benchmarks.
- Tasks: `segment`, `pose`, `obb`.
- Backends compared: `fallback` vs `flash_attn_turing`.

## Backend Detection
- Fallback suite backend: `fallback`
- Turing suite backend: `flash_attn_turing`
- Total fallback wall time (all tasks): `228.65s`
- Total turing wall time (all tasks): `214.62s`
- Overall speedup (fallback/turing): `1.0654x`

## Results

| Task | Batch | Fallback wall (s) | Turing wall (s) | Delta % (tu-fb)/fb | Speedup (fb/tu) | Metric key | Fallback metric | Turing metric |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| segment | 16 | 84.17 | 75.22 | -10.63% | 1.1189x | metrics/mAP50-95(M) | 0.0000 | 0.0000 |
| pose | 16 | 74.86 | 69.88 | -6.66% | 1.0713x | metrics/mAP50-95(P) | 0.0000 | 0.0000 |
| obb | 8 | 69.62 | 69.53 | -0.14% | 1.0014x | metrics/mAP50-95(B) | 0.0000 | 0.0000 |

## Visualizations
- Grouped wall time: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_wall_time_grouped.png`
- Grouped avg epoch time: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_avg_epoch_grouped.png`
- Avg epoch line comparison: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_avg_epoch_line_compare.png`
- Wall-time delta percent: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_wall_delta_pct.png`
- Speedup bars: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_speedup.png`
- Final primary metric grouped bars: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_primary_metric_grouped.png`
- Per-task metric curves: `/kaggle/working/phase2_l_flash_compare/plots/l_tasks_metric_curves.png`

## Artifact Root
- `/kaggle/working/phase2_l_flash_compare`
