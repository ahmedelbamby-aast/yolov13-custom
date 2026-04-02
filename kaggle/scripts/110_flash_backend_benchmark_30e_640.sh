#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PY="${Y13_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv. Run 10_setup_uv.sh first." >&2
  exit 1
fi

RUN_ROOT="${Y13_OUTPUT_DIR}/y13_bench_30e_640"
mkdir -p "${RUN_ROOT}"

DATA="${Y13_BENCH_DATA:-coco128.yaml}"
EPOCHS="${Y13_BENCH_EPOCHS:-30}"
IMGSZ="${Y13_BENCH_IMGSZ:-640}"
BATCH="${Y13_BENCH_BATCH:-8}"
WORKERS="${Y13_BENCH_WORKERS:-4}"

# Ensure turing backend is installed for the turing case
if [[ "${Y13_BENCH_ENSURE_TURING_INSTALL:-1}" == "1" ]]; then
  Y13_INSTALL_TURING_FLASH=1 bash "${SCRIPT_DIR}/25_install_turing_flash.sh"
fi

run_case() {
  local CASE_NAME="$1"
  local USE_TURING="$2"
  local DISABLE_FLASH="$3"

  echo "[bench] case=${CASE_NAME} use_turing=${USE_TURING} disable_flash=${DISABLE_FLASH}"

  Y13_USE_TURING_FLASH="${USE_TURING}" Y13_DISABLE_FLASH="${DISABLE_FLASH}" "${PY}" - <<'PY'
import csv
import json
import os
import time
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.modules import block

run_root = Path(os.environ['RUN_ROOT'])
case_name = os.environ['CASE_NAME']

data = os.environ['DATA']
epochs = int(os.environ['EPOCHS'])
imgsz = int(os.environ['IMGSZ'])
batch = int(os.environ['BATCH'])
workers = int(os.environ['WORKERS'])

start = time.perf_counter()
model = YOLO('ultralytics/cfg/models/v13/yolov13.yaml')
model.train(
    data=data,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    workers=workers,
    device='0,1',
    optimizer='AdamW',
    lr0=1e-3,
    weight_decay=5e-4,
    project=str(run_root),
    name=case_name,
    exist_ok=True,
    amp=True,
)
end = time.perf_counter()

run_dir = run_root / case_name
csv_path = run_dir / 'results.csv'

rows = []
if csv_path.exists():
    with csv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

final = rows[-1] if rows else {}
elapsed = end - start

payload = {
    'case': case_name,
    'flash_backend_selected': getattr(block, 'FLASH_BACKEND', 'unknown'),
    'use_flash_attn': bool(getattr(block, 'USE_FLASH_ATTN', False)),
    'flash_error': getattr(block, 'FLASH_ERROR', ''),
    'dataset': data,
    'epochs': epochs,
    'imgsz': imgsz,
    'batch': batch,
    'workers': workers,
    'wall_seconds': elapsed,
    'results_csv': str(csv_path),
    'final_metrics': final,
}

(run_root / f'{case_name}_metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps({'case': case_name, 'wall_seconds': round(elapsed, 2), 'backend': payload['flash_backend_selected']}))
PY
}

export RUN_ROOT DATA EPOCHS IMGSZ BATCH WORKERS

export CASE_NAME="flash_fallback"
run_case "flash_fallback" "0" "1"

export CASE_NAME="flash_turing"
run_case "flash_turing" "1" "0"

"${PY}" - <<'PY'
import json
from pathlib import Path

run_root = Path('/kaggle/working/y13_bench_30e_640')
fallback = json.loads((run_root / 'flash_fallback_metrics.json').read_text(encoding='utf-8'))
turing = json.loads((run_root / 'flash_turing_metrics.json').read_text(encoding='utf-8'))

fb = fallback['wall_seconds']
tu = turing['wall_seconds']
if tu > 0:
    speedup = fb / tu
else:
    speedup = 0.0

report = Path('/kaggle/work_here/yolov13/kaggle/reports/FLASH_BACKEND_BENCHMARK_30E_640.md')
lines = [
    '# Flash Backend Benchmark (30 Epochs, 640x640)',
    '',
    '## Setup',
    '',
    f"- Dataset: `{fallback['dataset']}`",
    f"- Epochs: `{fallback['epochs']}`",
    f"- Image size: `{fallback['imgsz']}`",
    f"- Batch: `{fallback['batch']}`",
    f"- Workers: `{fallback['workers']}`",
    '- Device: `0,1` (DDP on 2x T4)',
    '',
    '## Results',
    '',
    f"- Fallback backend wall time (s): `{fb:.2f}`",
    f"- Turing flash backend wall time (s): `{tu:.2f}`",
    f"- Speedup (fallback / turing): `{speedup:.4f}x`",
    '',
    '## Backend Detection',
    '',
    f"- Fallback case selected backend: `{fallback['flash_backend_selected']}`",
    f"- Turing case selected backend: `{turing['flash_backend_selected']}`",
    '',
    '## Artifact Paths',
    '',
    '- Run root: `/kaggle/working/y13_bench_30e_640`',
    '- JSON metrics: `/kaggle/working/y13_bench_30e_640/flash_fallback_metrics.json`',
    '- JSON metrics: `/kaggle/working/y13_bench_30e_640/flash_turing_metrics.json`',
]
report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(report)
PY

echo "Benchmark complete. Report: /kaggle/work_here/yolov13/kaggle/reports/FLASH_BACKEND_BENCHMARK_30E_640.md"
