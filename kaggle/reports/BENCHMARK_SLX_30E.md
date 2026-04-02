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
