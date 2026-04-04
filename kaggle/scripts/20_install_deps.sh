#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv. Run 10_setup_uv.sh first." >&2
  exit 1
fi

uv pip install --python "${PY}" -e "${Y13_ROOT}"
uv pip install --python "${PY}" rich

# Optional explicit torch stack overrides for reproducible benchmarking.
if [[ -n "${Y13_TORCH_VERSION:-}" ]]; then
  TORCH_SPEC="torch==${Y13_TORCH_VERSION}"
  if [[ -n "${Y13_TORCHVISION_VERSION:-}" ]]; then
    TORCH_SPEC+=" torchvision==${Y13_TORCHVISION_VERSION}"
  fi
  if [[ -n "${Y13_TORCH_EXTRA_INDEX_URL:-}" ]]; then
    uv pip install --python "${PY}" --extra-index-url "${Y13_TORCH_EXTRA_INDEX_URL}" ${TORCH_SPEC}
  else
    uv pip install --python "${PY}" ${TORCH_SPEC}
  fi
fi

"${PY}" - <<'PY'
from pathlib import Path
import subprocess
import sys

req = Path('requirements.txt')
if not req.exists():
    raise SystemExit('requirements.txt not found')

skip_tokens = ('flash_attn', '.whl', 'onnx==1.14.0', 'onnxruntime==1.15.1')
lines = [x.strip() for x in req.read_text(encoding='utf-8').splitlines() if x.strip() and not x.strip().startswith('#')]

for line in lines:
    if any(t in line for t in skip_tokens):
        print(f'[skip] {line}')
        continue
    cmd = ['uv', 'pip', 'install', '--python', sys.executable, line]
    print(f'[install] {line}')
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f'[warn] failed to install: {line}')
PY

if [[ "${Y13_INSTALL_TURING_FLASH:-0}" == "1" || "${Y13_USE_TURING_FLASH:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/25_install_turing_flash.sh"
fi

# Prefer cp312-compatible ONNX runtime stack
uv pip install --python "${PY}" onnx==1.17.0 onnxruntime-gpu==1.18.0 || true
