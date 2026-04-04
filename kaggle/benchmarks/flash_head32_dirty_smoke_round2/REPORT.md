# Dirty Smoke Round 2

| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train epoch (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0 | 10152 | 2.700 | 133.20 |
| head32 enabled | 99.76 | 10128 | 24 | 2.700 | 133.20 |

## Delta (enabled - baseline)

- flash hit-rate: 99.76 pp
- val inference: 0.000 ms/img
- train epoch time: 0.000 s
- speedup: 1.000x

## Fallback reasons

- baseline: `{'not_cuda': 24, 'unsupported_head_dim_32': 10128}`
- head32 enabled: `{'not_cuda': 24}`

## CUDA-only telemetry

- baseline cuda hit-rate: 0.00% (0/10128)
- head32 cuda hit-rate: 100.00% (10128/10128)
