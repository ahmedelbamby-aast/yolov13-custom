#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export DEBIAN_FRONTEND=noninteractive
sudo apt update
sudo apt install -y nvidia-driver-535

nvidia-smi || true
ls -l /dev/nvidia* || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

python - <<'PY2'
import torch
print("torch:", torch.__version__)
print("torch built with cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY2
