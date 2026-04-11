#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROBOFLOW_URL="${Y13_ROBOFLOW_URL:-https://app.roboflow.com/ds/xDP1PWcIeF?key=cojeRn8cUX}"
DATASETS_DIR="${Y13_DATASETS_DIR:-${Y13_WORKDIR}/datasets}"
DATASET_NAME="${Y13_ROBOFLOW_DATASET_NAME:-roboflow_custom_detect_dirty}"
TARGET_DIR="${DATASETS_DIR}/${DATASET_NAME}"
ZIP_PATH="${DATASETS_DIR}/${DATASET_NAME}.zip"
FORCE="${Y13_ROBOFLOW_FORCE:-0}"
KEEP_ZIP="${Y13_ROBOFLOW_KEEP_ZIP:-0}"
NORMALIZE_YAML="${Y13_ROBOFLOW_NORMALIZE_YAML:-1}"
REMAP_STUDENT_TEACHER="${Y13_ROBOFLOW_REMAP_STUDENT_TEACHER:-0}"  # backward-compatible shortcut
REMAP_ENABLE="${Y13_ROBOFLOW_REMAP_ENABLE:-0}"
REMAP_INCLUDE_NAMES="${Y13_ROBOFLOW_REMAP_INCLUDE_NAMES:-}"
REMAP_INCLUDE_IDS="${Y13_ROBOFLOW_REMAP_INCLUDE_IDS:-}"

mkdir -p "${DATASETS_DIR}"

if [[ -d "${TARGET_DIR}" && "${FORCE}" != "1" ]]; then
  echo "[roboflow_ready] dataset already exists: ${TARGET_DIR}"
  echo "[roboflow_ready] set Y13_ROBOFLOW_FORCE=1 to re-download"
  exit 0
fi

if [[ "${FORCE}" == "1" && -d "${TARGET_DIR}" ]]; then
  rm -rf "${TARGET_DIR}"
fi

echo "[roboflow_ready] downloading dataset from Roboflow"
curl -L "${ROBOFLOW_URL}" -o "${ZIP_PATH}"

mkdir -p "${TARGET_DIR}"

export Y13_ROBOFLOW_ZIP_PATH="${ZIP_PATH}"
export Y13_ROBOFLOW_TARGET_DIR="${TARGET_DIR}"
export Y13_ROBOFLOW_NORMALIZE_YAML="${NORMALIZE_YAML}"

python - <<'PY'
import json
import os
import zipfile
from pathlib import Path

import yaml

zip_path = Path(os.environ["Y13_ROBOFLOW_ZIP_PATH"])
target_dir = Path(os.environ["Y13_ROBOFLOW_TARGET_DIR"])

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_dir)

yaml_candidates = sorted(target_dir.rglob("data.yaml"), key=lambda p: len(p.parts))
if not yaml_candidates:
    raise SystemExit(f"[roboflow_ready] data.yaml not found under {target_dir}")

data_yaml = yaml_candidates[0]

if os.environ.get("Y13_ROBOFLOW_NORMALIZE_YAML", "1") == "1":
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d["path"] = str(target_dir)
        for key in ("train", "val", "valid", "test"):
            v = d.get(key)
            if isinstance(v, str):
                while v.startswith("../"):
                    v = v[3:]
                if v.startswith("./"):
                    v = v[2:]
                d[key] = v
        data_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")

report = {
    "target_dir": str(target_dir),
    "data_yaml": str(data_yaml),
    "zip_path": str(zip_path),
}

meta_path = target_dir / "_roboflow_ready.json"
meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"[roboflow_ready] extracted to: {target_dir}")
print(f"[roboflow_ready] detected data.yaml: {data_yaml}")
print(f"[roboflow_ready] metadata: {meta_path}")
PY

unset Y13_ROBOFLOW_ZIP_PATH Y13_ROBOFLOW_TARGET_DIR Y13_ROBOFLOW_NORMALIZE_YAML

if [[ "${REMAP_STUDENT_TEACHER}" == "1" && -z "${REMAP_INCLUDE_NAMES}" && -z "${REMAP_INCLUDE_IDS}" ]]; then
  REMAP_ENABLE=1
  REMAP_INCLUDE_NAMES="student,teacher"
fi

if [[ -n "${REMAP_INCLUDE_NAMES}" || -n "${REMAP_INCLUDE_IDS}" ]]; then
  REMAP_ENABLE=1
fi

if [[ "${REMAP_ENABLE}" == "1" ]]; then
  DATA_YAML_PATH="${TARGET_DIR}/data.yaml"
  REMAP_CMD=(python "${SCRIPT_DIR}/38_class_remap.py" --data "${DATA_YAML_PATH}")

  if [[ -n "${REMAP_INCLUDE_IDS}" ]]; then
    IFS=',' read -r -a remap_ids <<< "${REMAP_INCLUDE_IDS}"
    for cid in "${remap_ids[@]}"; do
      cid="${cid//[[:space:]]/}"
      [[ -z "${cid}" ]] && continue
      REMAP_CMD+=(--include-id "${cid}")
    done
  fi

  if [[ -n "${REMAP_INCLUDE_NAMES}" ]]; then
    IFS=',' read -r -a remap_names <<< "${REMAP_INCLUDE_NAMES}"
    for cname in "${remap_names[@]}"; do
      cname="${cname#"${cname%%[![:space:]]*}"}"
      cname="${cname%"${cname##*[![:space:]]}"}"
      [[ -z "${cname}" ]] && continue
      REMAP_CMD+=(--include-name "${cname}")
    done
  fi

  if [[ "${#REMAP_CMD[@]}" -le 4 ]]; then
    echo "[roboflow_ready] remap requested, but no classes were provided." >&2
    echo "[roboflow_ready] use Y13_ROBOFLOW_REMAP_INCLUDE_NAMES or Y13_ROBOFLOW_REMAP_INCLUDE_IDS" >&2
    exit 1
  fi

  "${REMAP_CMD[@]}"
fi

if [[ "${KEEP_ZIP}" != "1" ]]; then
  rm -f "${ZIP_PATH}"
fi

echo "[roboflow_ready] done"
