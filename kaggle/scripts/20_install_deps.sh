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

"${PY}" - <<'PY'
from pathlib import Path
import subprocess
import sys

req = Path('requirements.txt')
if not req.exists():
    raise SystemExit('requirements.txt not found')

skip_tokens = ('flash_attn', '.whl')
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
