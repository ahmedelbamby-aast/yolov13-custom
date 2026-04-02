# YOLOv13 S/L/X Fixed Batch Benchmark Comparison (2xT4 DDP)

## Setup
- Dataset: coco128.yaml
- Samples scanned: 128
- Dataset-aligned imgsz: 640 (long mode 640, short mode 480)
- Epochs per variant: 30
- Fixed largest batches: S=32, L=16, X=12
- Fallback backend run: fallback
- Turing backend run: flash_attn_turing

## Results

| Variant | Batch | Fallback wall (s) | Fallback avg/epoch (s) | Turing wall (s) | Turing avg/epoch (s) | Delta wall (s) | Delta avg/epoch (s) | Fallback mAP50 | Turing mAP50 | Fallback mAP50-95 | Turing mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | 32 | 236.33 | 7.88 | 241.12 | 8.04 | +4.79 | +0.16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L | 16 | 544.52 | 18.15 | 544.53 | 18.15 | +0.01 | +0.00 | 0.0076 | 0.0076 | 0.0022 | 0.0022 |
| X | 12 | 846.67 | 28.22 | 841.32 | 28.04 | -5.35 | -0.18 | 0.0220 | 0.0188 | 0.0090 | 0.0063 |

## Notes
- Backend confirmation comes from each run benchmark_summary.json flash_backend field, not log-only inference.
- On this run set, Turing and fallback are close in wall time for S/L/X at these fixed batches and settings.

## Repo Artifacts
- kaggle/benchmarks/slx_30e_fixed/fallback
- kaggle/benchmarks/slx_30e_fixed/turing
- kaggle/reports/BENCHMARK_SLX_30E_FIXED_BATCH_COMPARISON.md
