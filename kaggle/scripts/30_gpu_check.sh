#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"

"${PY}" - <<'PY'
import os
import torch

print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES'))
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

if torch.cuda.device_count() < 2:
    raise SystemExit('Expected 2 GPUs for DDP target setup')
PY
