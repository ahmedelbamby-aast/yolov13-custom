#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found" >&2
  exit 1
fi

if [[ ! -x "${Y13_ROOT}/.venv/bin/python" ]]; then
  uv venv "${Y13_ROOT}/.venv" --python /usr/bin/python3 --system-site-packages
else
  echo "Using existing virtual environment: ${Y13_ROOT}/.venv"
fi

uv pip install --python "${Y13_ROOT}/.venv/bin/python" --upgrade pip setuptools wheel

"${Y13_ROOT}/.venv/bin/python" -V

ACTIVATE_PATH="${Y13_ROOT}/.venv/bin/activate"
MARKER="# Y13: ensure torch CUDA shared libs are discoverable (turFlash import)"

if [[ -f "${ACTIVATE_PATH}" ]] && ! grep -Fq "${MARKER}" "${ACTIVATE_PATH}"; then
  cat <<'EOF' >> "${ACTIVATE_PATH}"

# Y13: ensure torch CUDA shared libs are discoverable (turFlash import)
if [ -n "${VIRTUAL_ENV:-}" ]; then
  for _y13_torch_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib; do
    if [ -d "$_y13_torch_lib" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$_y13_torch_lib:"*) ;;
        *) export LD_LIBRARY_PATH="$_y13_torch_lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
      break
    fi
  done
  unset _y13_torch_lib
fi
EOF
fi
