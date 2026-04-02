# Flash Backend Benchmark (30 Epochs, 640x640)

## Setup

- Dataset: `coco128.yaml`
- Epochs: `30`
- Image size: `640`
- Batch: `8`
- Workers: `4`
- Device: `0,1` (DDP on 2x T4)

## Results

- Fallback backend wall time (s): `271.41`
- Turing flash backend wall time (s): `265.62`
- Speedup (fallback / turing): `1.0218x`

## Backend Detection

- Fallback case selected backend: `fallback`
- Turing case selected backend: `flash_attn_turing`

## Artifact Paths

- Run root: `/kaggle/working/y13_bench_30e_640`
- JSON metrics: `/kaggle/working/y13_bench_30e_640/flash_fallback_metrics.json`
- JSON metrics: `/kaggle/working/y13_bench_30e_640/flash_turing_metrics.json`


## Visualizations

- Total runtime bar chart: `/kaggle/working/y13_bench_30e_640/plots/flash_runtime_bar.png`
- Per-epoch runtime plot: `/kaggle/working/y13_bench_30e_640/plots/flash_runtime_per_epoch.png`
