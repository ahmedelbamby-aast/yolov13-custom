# Kaggle Notebooks (Ultralytics API Style, No SSH)

These notebooks use normal Ultralytics usage style:
- `YOLO(...).train(...)`
- `model.val(...)`
- `model.export(...)`
- `model.track(...)`
- `model.predict(...)`

No project shell script calls are used inside notebook workflows.

- `06_benchmark_flash.ipynb`: S/L/X fixed-largest-batch comparison (Turing flash vs fallback) with runtime and avg-epoch plots
