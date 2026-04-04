# Dirty Smoke Round 1

| Run | Flash hit rate (%) | Hits | Fallbacks | Val inference (ms/img) | Train epoch (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0 | 10152 | 3.000 | 118.08 |
| head32 enabled | 99.76 | 10128 | 24 | 2.500 | 115.92 |

## Delta (enabled - baseline)

- flash hit-rate: 99.76 pp
- val inference: -0.500 ms/img
- train epoch time: -2.160 s
- speedup: 1.019x

## Fallback reasons

- baseline: `{'not_cuda': 24, 'unsupported_head_dim_32': 10128}`
- head32 enabled: `{'not_cuda': 24}`
