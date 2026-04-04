# turFlash head_dim=32 dirty-data smoke comparison

- baseline: `Y13_ENABLE_TURING_HEAD_DIM32=0`
- enabled: `Y13_ENABLE_TURING_HEAD_DIM32=1`
- parallel policy: baseline gpu=1, head32 gpu=0, workers per run=2
- epochs: 5
- dataset: remapped dirty Roboflow subset (`student`, `teacher`)
- fraction: 0.05
- train cache: ram

## Results

| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train epoch (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0 | 10152 | 2.70 | 133.20 |
| head32 enabled | 99.76 | 10128 | 24 | 2.70 | 133.20 |

## CUDA-only telemetry

- baseline cuda hit-rate: 0.00% (0/10128)
- head32 cuda hit-rate: 100.00% (10128/10128)

## Delta (enabled - baseline)

- flash hit-rate: +99.76 percentage points
- val inference: +0.000 ms/img
- train epoch time: +0.000 s
- train speedup: 1.00x

## Fallback reasons

- baseline: `{'not_cuda': 24, 'unsupported_head_dim_32': 10128}`
- head32 enabled: `{'not_cuda': 24}`

Artifacts:

- `kaggle/benchmarks/flash_head32_dirty_smoke/compare_summary.json`
- `kaggle/benchmarks/flash_head32_dirty_smoke/telemetry_compare.png`
- `kaggle/benchmarks/flash_head32_dirty_smoke/REPORT.md`
