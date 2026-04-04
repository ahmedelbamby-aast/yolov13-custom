# turFlash head_dim=32 smoke comparison

- baseline: `Y13_ENABLE_TURING_HEAD_DIM32=0` (default)
- enabled: `Y13_ENABLE_TURING_HEAD_DIM32=1`
- model/data: `yolov13n` + `coco8.yaml`, 1 epoch, imgsz 320, batch 8, device 0

## Results

| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train iter (s/it) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0 | 144 | 9.20 | n/a |
| head32 enabled | 83.33 | 120 | 24 | 7.70 | n/a |

## Delta (enabled - baseline)

- flash hit-rate: +83.33 percentage points
- val inference: -1.500 ms/img
- train iter: n/a s/it

## Fallback reasons

- baseline: `{'not_cuda': 24, 'unsupported_head_dim_32': 120}`
- head32 enabled: `{'not_cuda': 24}`

Artifacts:

- `kaggle/benchmarks/flash_head32_smoke/compare_summary.json`
- `kaggle/benchmarks/flash_head32_smoke/telemetry_compare.png`
- `kaggle/benchmarks/flash_head32_smoke/REPORT.md`
