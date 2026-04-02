#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ls -l /dev/nvidia* || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python - <<'PY2'
import os
print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch built with cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch error:", e)
PY2
