#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export Y13_ROOT="${Y13_ROOT:-${REPO_ROOT_DEFAULT}}"
export Y13_WORKDIR="${Y13_WORKDIR:-/kaggle/work_here}"
export Y13_OUTPUT_DIR="${Y13_OUTPUT_DIR:-/kaggle/working}"
export Y13_VENV="${Y13_VENV:-${Y13_ROOT}/.venv}"
export Y13_DEVELOPER_APPROVED_DESTRUCTIVE="${Y13_DEVELOPER_APPROVED_DESTRUCTIVE:-0}"
export Y13_AUTO_FLASH_INSTALL="${Y13_AUTO_FLASH_INSTALL:-1}"

y13_assert_server_safe() {
  local action="${1:-}"
  case "${action}" in
    reboot|shutdown|delete_server|delete_system_files|format_disk)
      if [[ "${Y13_DEVELOPER_APPROVED_DESTRUCTIVE}" != "1" ]]; then
        echo "[y13-safety] blocked destructive action '${action}'. Inform developer and set Y13_DEVELOPER_APPROVED_DESTRUCTIVE=1 only with explicit approval." >&2
        return 1
      fi
      ;;
  esac
  return 0
}

y13_detect_gpu_names() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' || true
  fi
}

y13_auto_flash_mode() {
  local gpu_names
  gpu_names="$(y13_detect_gpu_names)"
  if [[ -z "${gpu_names}" ]]; then
    echo "fallback"
    return
  fi
  if echo "${gpu_names}" | grep -qi 't4'; then
    echo "turing"
    return
  fi
  echo "auto"
}

# Flash backend controls
export Y13_USE_TURING_FLASH="${Y13_USE_TURING_FLASH:-0}"
export Y13_INSTALL_TURING_FLASH="${Y13_INSTALL_TURING_FLASH:-0}"
export Y13_DISABLE_FLASH="${Y13_DISABLE_FLASH:-0}"

if [[ -z "${Y13_FLASH_MODE:-}" ]]; then
  export Y13_FLASH_MODE="$(y13_auto_flash_mode)"
fi

if [[ "${Y13_FLASH_MODE}" == "turing" && "${Y13_AUTO_FLASH_INSTALL}" == "1" && "${Y13_INSTALL_TURING_FLASH}" != "1" ]]; then
  export Y13_INSTALL_TURING_FLASH=1
fi

# Canonical flash mode precedence:
# 1) Y13_FLASH_MODE (fallback|turing|flash4|auto)
# 2) explicit legacy env flags above
# 3) default fallback to turing for Kaggle workflows
if [[ -n "${Y13_FLASH_MODE:-}" ]]; then
  case "${Y13_FLASH_MODE}" in
    fallback)
      export Y13_DISABLE_FLASH=1
      export Y13_USE_TURING_FLASH=0
      export Y13_PREFER_FLASH4=0
      ;;
    turing)
      export Y13_DISABLE_FLASH=0
      export Y13_USE_TURING_FLASH=1
      export Y13_PREFER_FLASH4=0
      ;;
    flash4)
      export Y13_DISABLE_FLASH=0
      export Y13_USE_TURING_FLASH=0
      export Y13_PREFER_FLASH4=1
      ;;
    auto)
      export Y13_DISABLE_FLASH=0
      export Y13_USE_TURING_FLASH=0
      export Y13_PREFER_FLASH4=0
      ;;
  esac
fi

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if [[ -d "${Y13_VENV}" ]]; then
  for y13_torch_lib in "${Y13_VENV}"/lib/python*/site-packages/torch/lib; do
    if [[ -d "${y13_torch_lib}" ]]; then
      case ":${LD_LIBRARY_PATH}:" in
        *":${y13_torch_lib}:"*) ;;
        *) export LD_LIBRARY_PATH="${y13_torch_lib}:${LD_LIBRARY_PATH}" ;;
      esac
      break
    fi
  done
  unset y13_torch_lib
fi

export PYTHONPATH="${Y13_ROOT}:${PYTHONPATH:-}"

mkdir -p "${Y13_WORKDIR}" "${Y13_OUTPUT_DIR}"
cd "${Y13_ROOT}"
