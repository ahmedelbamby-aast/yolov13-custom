#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found" >&2
  exit 1
fi

if [[ "${Y13_FRESH_VENV:-0}" == "1" && -d "${Y13_ROOT}/.venv" ]]; then
  echo "[setup] removing existing virtual environment for fresh install: ${Y13_ROOT}/.venv"
  rm -rf "${Y13_ROOT}/.venv"
fi

PYTHON_BIN="${Y13_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "/usr/bin/python3" ]]; then
    PYTHON_BIN="/usr/bin/python3"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "python interpreter not found" >&2
    exit 1
  fi
fi

if [[ ! -x "${Y13_ROOT}/.venv/bin/python" ]]; then
  uv venv "${Y13_ROOT}/.venv" --python "${PYTHON_BIN}" --system-site-packages
else
  echo "Using existing virtual environment: ${Y13_ROOT}/.venv"
fi

VENV_PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" && -x "${Y13_ROOT}/.venv/Scripts/python.exe" ]]; then
  VENV_PY="${Y13_ROOT}/.venv/Scripts/python.exe"
fi
if [[ ! -x "${VENV_PY}" ]]; then
  echo "virtual environment python not found after creation" >&2
  exit 1
fi

uv pip install --python "${VENV_PY}" --upgrade pip setuptools wheel

"${VENV_PY}" -V

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
