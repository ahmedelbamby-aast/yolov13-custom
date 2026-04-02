# YOLOv13 S/L/X Benchmark (DDP 2xT4)

## Dataset / imgsz alignment
- Dataset: `coco128.yaml`
- Samples checked: `128`
- Long side mode: `640`
- Short side mode: `480`
- Selected imgsz: `640`

## Flash backend
- `fallback`

## Results
- S: batch `32`, wall `232.84s`, mAP50 `0.0000`, mAP50-95 `0.0000`
- L: batch `16`, wall `544.79s`, mAP50 `0.0076`, mAP50-95 `0.0022`
- X: batch `12`, wall `841.40s`, mAP50 `0.0220`, mAP50-95 `0.0090`

## Artifacts
- `/kaggle/working/y13_bench_slx_30e/plots/slx_runtime_bar.png`
- `/kaggle/working/y13_bench_slx_30e/plots/slx_map50_line.png`
- `/kaggle/working/y13_bench_slx_30e/feature_maps`

## Fixed Largest Batch Follow-up (S=32, L=16, X=12)

A follow-up benchmark was executed with fixed largest stable batches and explicit backend A/B.

- Fallback run backend: `fallback`
- Turing run backend: `flash_attn_turing`
- Detailed comparison report: `kaggle/reports/BENCHMARK_SLX_30E_FIXED_BATCH_COMPARISON.md`
- Synced artifacts: `kaggle/benchmarks/slx_30e_fixed`
