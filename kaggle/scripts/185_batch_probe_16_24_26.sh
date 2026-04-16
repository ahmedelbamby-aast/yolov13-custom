#!/usr/bin/env bash
set -u

cd "${Y13_ROOT:-$HOME/kaggle/work_here/yolov13}"
source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUT="${Y13_ARCH_COMPARE_OUT:-$HOME/kaggle/working/arch_compare_latest}"
DATA="${Y13_ARCH_COMPARE_DATA:-$HOME/kaggle/work_here/dataset/converted_dataset/data.yaml}"
mkdir -p "$OUT"

STATUS_CSV="$OUT/batch_probe_status_16_24_26.csv"
SUMMARY_TXT="$OUT/batch_probe_summary_16_24_26.txt"
RUN_LOG="$OUT/batch_probe_run_16_24_26.log"

echo "batch,rc" > "$STATUS_CSV"
echo "[batch-probe] start $(date -Is)" | tee "$RUN_LOG"

for B in 16 24 26; do
  NAME="telemetry_flash_on_3e_b${B}_probe"
  LOG="$OUT/${NAME}.log"
  echo "[batch-probe] batch=$B start $(date -Is)" | tee -a "$RUN_LOG"
  python scripts/train.py \
    --model ultralytics/cfg/models/v13/yolov13l.yaml \
    --data "$DATA" \
    --epochs 3 \
    --imgsz 240 \
    --batch "$B" \
    --device 0 \
    --workers 24 \
    --project "$OUT" \
    --name "$NAME" \
    --flash-mode auto \
    --arg fraction=0.50 \
    --arg prefetch_factor=8 \
    --arg persistent_workers=true \
    --arg classes=[0,9] \
    --arg plots=false \
    --arg exist_ok=true > "$LOG" 2>&1
  RC=$?
  echo "$B,$RC" >> "$STATUS_CSV"
  echo "[batch-probe] batch=$B rc=$RC end $(date -Is)" | tee -a "$RUN_LOG"
done

{
  echo "Batch probe summary"
  echo "generated_at=$(date -Is)"
  while IFS=, read -r b rc; do
    if [[ "$b" == "batch" ]]; then
      continue
    fi
    if [[ "$rc" == "0" ]]; then
      echo "batch=$b status=ok"
    else
      echo "batch=$b status=failed"
    fi
  done < "$STATUS_CSV"
} > "$SUMMARY_TXT"

echo "[batch-probe] done $(date -Is)" | tee -a "$RUN_LOG"
