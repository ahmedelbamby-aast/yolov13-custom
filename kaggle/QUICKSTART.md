# YOLOv13 Kaggle Quickstart (2x T4 DDP)

## Paths
- Development workspace: `/kaggle/work_here/yolov13`
- Outputs and artifacts: `/kaggle/working`
- Remote sync target: `github.com/ahmedelbamby-aast/yolov13-custom`

## End-to-end pipeline
```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/run_all.sh
```

This pipeline will:
1. Show a colorful Rich startup banner for Eng.Ahmed ElBamby.
2. Create a uv-based virtual environment.
3. Install core dependencies with edge-case handling.
4. Validate 2-GPU CUDA visibility.
5. Run a DDP smoke training job.
6. Package the project as `/kaggle/working/yolov13.zip`.

## Optional Flash Backend Flags (T4/Turing)
```bash
# default runtime uses Turing flash on T4
export Y13_INSTALL_TURING_FLASH=1   # one-time install/build
# export Y13_USE_TURING_FLASH=0     # disable if needed
# export Y13_DISABLE_FLASH=1        # force fallback backend
```

## Kaggle Notebook UI (No SSH)

Use prebuilt notebooks in `notebooks/`:
- `notebooks/01_train.ipynb`
- `notebooks/02_validate.ipynb`
- `notebooks/03_export.ipynb`
- `notebooks/04_tracking.ipynb`
- `notebooks/05_test.ipynb`

All notebooks use one canonical environment in `/kaggle/work_here/yolov13/.venv`.

## Packaging only
```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/50_package_zip.sh
```

## New machine GPU bring-up (2x T4)

For a fresh Kaggle machine session, run:



Manual equivalent commands:



If CUDA appears unavailable in Python while  exists, run:



## New machine GPU bring-up (2x T4)

For a fresh Kaggle machine session, run:

```bash
cd /kaggle/work_here/yolov13
bash kaggle/scripts/27_install_nvidia_driver_535.sh
bash kaggle/scripts/32_cuda_sanity_report.sh
```

Manual equivalent commands:

```bash
sudo apt update
sudo apt install -y nvidia-driver-535

ls -l /dev/nvidia*
echo $CUDA_VISIBLE_DEVICES
python - << PY
import torch
print("torch cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

If CUDA appears unavailable in Python while `/dev/nvidia*` exists, run:

```bash
apt-get update -y && apt-get install -y pciutils
lspci | grep -i nvidia || true
cat /proc/driver/nvidia/version || true
cat /proc/driver/nvidia/gpus/0000:00:04.0/information 2>/dev/null || true
cat /proc/driver/nvidia/gpus/0000:00:05.0/information 2>/dev/null || true
python - << PY
import os
print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
try:
    import tensorflow as tf
    print("TF GPUs:", tf.config.list_physical_devices(GPU))
except Exception as e:
    print("TF error:", e)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda avail:", torch.cuda.is_available())
    print("count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch error:", e)
PY
```
