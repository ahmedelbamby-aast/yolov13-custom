#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/bamby/kaggle/work_here/yolov13}"
PROJECT_DIR="${PROJECT_DIR:-/home/bamby/working/yolov13_l_p2}"
BASE_RUN="${BASE_RUN:-stress_test_i768_200e}"
BASE_DIR="${PROJECT_DIR}/${BASE_RUN}"
BASE_BEST="${BASE_DIR}/weights/best.pt"
DATA_YAML="${DATA_YAML:-/home/bamby/kaggle/work_here/dataset/converted_dataset/data.yaml}"

EPOCHS="${EPOCHS:-200}"
IMGSZ="${IMGSZ:-768}"
BATCH="${BATCH:-0.70}"

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export Y13_AMP_PROBE_MODEL="yolov13n.pt"
export Y13_DISABLE_FLASH="1"
export Y13_USE_TURING_FLASH="0"
export Y13_PREFER_FLASH4="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${BASE_BEST}" ]]; then
  echo "[pipeline] missing base checkpoint: ${BASE_BEST}" >&2
  exit 1
fi

if [[ ! -f "${DATA_YAML}" ]]; then
  echo "[pipeline] missing data yaml: ${DATA_YAML}" >&2
  exit 1
fi

python3 - <<'PY'
from pathlib import Path
import yaml

args_path = Path('/home/bamby/working/yolov13_l_p2/stress_test_i768_200e/args.yaml')
if args_path.exists():
    data = yaml.safe_load(args_path.read_text()) or {}
    data['imgsz'] = 768
    args_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    print(f"[pipeline] patched {args_path} imgsz=768")
else:
    print(f"[pipeline] args file not found, skipping patch: {args_path}")
PY

echo "[pipeline] stage-1 resume start: ${BASE_RUN}"
.venv/bin/python scripts/train.py \
  --model "${BASE_BEST}" \
  --data "${DATA_YAML}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device 0 \
  --workers 24 \
  --project "${PROJECT_DIR}" \
  --name "${BASE_RUN}" \
  --resume \
  --flash-mode fallback \
  --auto-flash-fallback

echo "[pipeline] stage-1 resume finished"

mapfile -t CONTRAST_ROWS < <(python3 - <<'PY'
import re
import yaml

data = yaml.safe_load(open('/home/bamby/kaggle/work_here/dataset/converted_dataset/data.yaml', 'r', encoding='utf-8'))
names = data.get('names', [])
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]

def slugify(text: str) -> str:
    text = text.strip().lower().replace(' ', '_')
    text = re.sub(r'[^a-z0-9_]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'class'

# Build contrast pairs among non-teacher classes only.
# Keep explicit semantic pairs first, then any remaining classes are paired sequentially.
preferred_pairs = [
    (7, 8),  # student_sitting vs student_standing
    (4, 5),  # student_looking_left vs student_looking_right
    (6, 2),  # student_looking_up vs student_looking_down
    (3, 1),  # student_looking_forward vs student_looking_backward
]

valid_ids = [i for i in range(len(names)) if i not in (0, 9)]
used = set()

for a, b in preferred_pairs:
    if a in valid_ids and b in valid_ids and a not in used and b not in used:
        used.add(a)
        used.add(b)
        print(f"{a},{b}\t{slugify(str(names[a]))}\t{slugify(str(names[b]))}")

remaining = [i for i in valid_ids if i not in used]
for i in range(0, len(remaining) - 1, 2):
    a, b = remaining[i], remaining[i + 1]
    print(f"{a},{b}\t{slugify(str(names[a]))}\t{slugify(str(names[b]))}")
PY
)

for row in "${CONTRAST_ROWS[@]}"; do
  pair_ids="${row%%$'\t'*}"
  rest="${row#*$'\t'}"
  slug_a="${rest%%$'\t'*}"
  slug_b="${rest#*$'\t'}"
  run_name="contrast_${slug_a}_vs_${slug_b}_i768"
  class_arg="classes=[${pair_ids}]"

  echo "[pipeline] stage-2 run start: ${run_name} (${class_arg})"
  .venv/bin/python scripts/train.py \
    --model "${BASE_BEST}" \
    --data "${DATA_YAML}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device 0 \
    --workers 24 \
    --project "${PROJECT_DIR}" \
    --name "${run_name}" \
    --flash-mode fallback \
    --auto-flash-fallback \
    --arg task=detect \
    --arg "${class_arg}" \
    --arg fraction=1 \
    --arg cache=ram \
    --arg prefetch_factor=8 \
    --arg persistent_workers=true \
    --arg compile=false \
    --arg profile=false \
    --arg amp=true \
    --arg deterministic=false \
    --arg val=true \
    --arg plots=true \
    --arg show=true \
    --arg visualize=true \
    --arg exist_ok=true \
    --arg close_mosaic=0 \
    --arg warmup_epochs=8 \
    --arg cos_lr=true \
    --arg lr0=0.02 \
    --arg lrf=0.10 \
    --arg box=7.5 \
    --arg cls=0.5 \
    --arg dfl=1.5 \
    --arg nbs=64 \
    --arg cls_pw=0.60
  echo "[pipeline] stage-2 run done: ${run_name}"
done

echo "[pipeline] all runs completed"
