#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" && -x "${Y13_ROOT}/.venv/Scripts/python.exe" ]]; then
  PY="${Y13_ROOT}/.venv/Scripts/python.exe"
fi
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv. Run 10_setup_uv.sh first." >&2
  exit 1
fi

uv pip install --python "${PY}" -e "${Y13_ROOT}"
uv pip install --python "${PY}" rich

# Torch stack defaults for turFlash-tested baseline (override via env vars if needed).
Y13_TORCH_VERSION="${Y13_TORCH_VERSION:-2.8.0}"
Y13_TORCHVISION_VERSION="${Y13_TORCHVISION_VERSION:-0.23.0}"
TORCH_SPEC="torch==${Y13_TORCH_VERSION} torchvision==${Y13_TORCHVISION_VERSION}"
if [[ -n "${Y13_TORCH_EXTRA_INDEX_URL:-}" ]]; then
  uv pip install --python "${PY}" --extra-index-url "${Y13_TORCH_EXTRA_INDEX_URL}" ${TORCH_SPEC}
else
  uv pip install --python "${PY}" ${TORCH_SPEC}
fi

# Keep NCCL runtime aligned with torch 2.8 cu12 wheels.
uv pip install --python "${PY}" "nvidia-nccl-cu12==${Y13_NCCL_VERSION:-2.27.3}"

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

if [[ "${Y13_AUTO_FLASH_INSTALL:-1}" == "1" ]]; then
  "${PY}" - <<'PY'
import importlib
import json
import os
from pathlib import Path

mods = ["flash_attn_turing", "flash_attn"]
status = {"selected": None, "imports": {}}
for m in mods:
    try:
        importlib.import_module(m)
        status["imports"][m] = "ok"
        if status["selected"] is None:
            status["selected"] = m
    except Exception as e:
        status["imports"][m] = f"fail: {e}"

out = Path(os.environ.get("Y13_OUTPUT_DIR", "/kaggle/working")) / "flash_backend_probe.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(f"[flash-probe] saved={out}")
print(json.dumps(status, indent=2))
PY
fi

# Prefer cp312-compatible ONNX runtime stack
uv pip install --python "${PY}" onnx==1.17.0 onnxruntime-gpu==1.18.0 || true
