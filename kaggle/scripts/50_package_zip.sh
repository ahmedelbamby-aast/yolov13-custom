#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

"${Y13_ROOT}/.venv/bin/python" - <<'PY'
from pathlib import Path
import zipfile

root = Path('/kaggle/work_here/yolov13')
out = Path('/kaggle/working/yolov13.zip')

exclude_parts = {'.git', '.venv', '__pycache__', '.pytest_cache'}
exclude_suffixes = {'.pt', '.pth', '.pyc'}

if out.exists():
    out.unlink()

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        if any(part in exclude_parts for part in rel.parts):
            continue
        if p.is_file() and p.suffix in exclude_suffixes:
            continue
        zf.write(p, rel.as_posix())

print(f'Created: {out}')
PY
