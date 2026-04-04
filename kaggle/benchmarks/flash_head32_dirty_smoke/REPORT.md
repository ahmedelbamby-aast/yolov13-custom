# turFlash head_dim=32 dirty-data smoke comparison

- baseline: `Y13_ENABLE_TURING_HEAD_DIM32=0`
- enabled: `Y13_ENABLE_TURING_HEAD_DIM32=1`
- epochs: 5
- dataset: remapped dirty Roboflow subset (`student`, `teacher`)
- fraction: 0.05

## Results

| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train epoch (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0 | 10152 | 3.00 | 118.08 |
| head32 enabled | 99.76 | 10128 | 24 | 2.50 | 115.92 |

## Delta (enabled - baseline)

- flash hit-rate: +99.76 percentage points
- val inference: -0.500 ms/img
- train epoch time: -2.160 s
- train speedup: 1.02x

## Fallback reasons

- baseline: `{'not_cuda': 24, 'unsupported_head_dim_32': 10128}`
- head32 enabled: `{'not_cuda': 24}`

Artifacts:

- `kaggle/benchmarks/flash_head32_dirty_smoke/compare_summary.json`
- `kaggle/benchmarks/flash_head32_dirty_smoke/telemetry_compare.png`
- `kaggle/benchmarks/flash_head32_dirty_smoke/REPORT.md`
