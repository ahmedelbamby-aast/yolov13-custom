#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found" >&2
  exit 1
fi

uv venv "${Y13_ROOT}/.venv" --python /usr/bin/python3 --system-site-packages
uv pip install --python "${Y13_ROOT}/.venv/bin/python" --upgrade pip setuptools wheel

"${Y13_ROOT}/.venv/bin/python" -V
