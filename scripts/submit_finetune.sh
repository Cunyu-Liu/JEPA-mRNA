#!/bin/bash
# Submit one fine-tuning job: pick the emptiest GPU, run it, survive OOM.
#
# This cluster is a shared node with no scheduler, and other users allocate
# concurrently, so a card that looks free at submit time can fill up mid-run.
# The submitter therefore:
#   1. picks the full (non-MIG) GPU with the most free memory, requiring a
#      caller-specified minimum;
#   2. on CUDA OOM, retries on the next-best GPU that has not been tried;
#   3. if every card OOMs, halves the batch size and retries once more
#      (recorded in the ledger as a protocol deviation);
#   4. appends a ledger row for every attempt so the experiment bookkeeping is
#      never reconstructed from memory.
#
# Usage:
#   submit_finetune.sh --task NAME --task_kind K --model_path P --data_dir D \
#                      --out_dir O [--seed S] [--max_len N] [--batch B] \
#                      [--epochs E] [--lr LR] [--min_free MIB] [--fp16] \
#                      [--save_model] [--grad_checkpoint] [--python PY] [--tag T]
set -uo pipefail

TASK=""; TASK_KIND=""; MODEL_PATH=""; DATA_DIR=""; OUT_DIR=""
SEED=42; MAX_LEN=512; BATCH=8; EVAL_BATCH=32; EPOCHS=20; LR=1e-4; MIN_FREE=16000
FP16=0; SAVE_MODEL=0; GRAD_CKPT=0; FREEZE=0; ENC_LR_SCALE=1.0
PYTHON="${RNAJEPA_PYTHON:-/home/cunyuliu/miniconda3/envs/mrnabert/bin/python}"
TAG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --task_kind) TASK_KIND="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --max_len) MAX_LEN="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --eval_batch) EVAL_BATCH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --min_free) MIN_FREE="$2"; shift 2;;
    --fp16) FP16=1; shift;;
    --save_model) SAVE_MODEL=1; shift;;
    --grad_checkpoint) GRAD_CKPT=1; shift;;
    --freeze_encoder) FREEZE=1; shift;;
    --encoder_lr_scale) ENC_LR_SCALE="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

for v in TASK TASK_KIND MODEL_PATH DATA_DIR OUT_DIR; do
  if [ -z "${!v}" ]; then echo "missing required --$(echo $v | tr 'A-Z_' 'a-z-')" >&2; exit 2; fi
done

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
LEDGER="${RNAJEPA_LEDGER:-/mnt/cunyuliu/rna-jepa/ledger.jsonl}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"

export PYTHONPATH="${RNAJEPA_PYEXT:-/mnt/cunyuliu/rna-jepa/pyext}:$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

source "$ROOT/scripts/gpu_util.sh"

ledger_row() {  # status device attempt batch
  local status="$1" device="$2" attempt="$3" batch="$4"
  python3 - "$LEDGER" <<PYEOF
import json, os, socket, sys, datetime
row = {
    "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "task": "$TASK", "task_kind": "$TASK_KIND", "tag": "$TAG",
    "model_path": "$MODEL_PATH", "data_dir": "$DATA_DIR", "out_dir": "$OUT_DIR",
    "seed": $SEED, "max_len": $MAX_LEN, "batch": $batch,
    "epochs": $EPOCHS, "lr": $LR,
    "fp16": bool($FP16), "save_model": bool($SAVE_MODEL),
    "grad_checkpoint": bool($GRAD_CKPT), "freeze_encoder": bool($FREEZE), "encoder_lr_scale": $ENC_LR_SCALE,
    "device": "$device", "attempt": "$attempt", "status": "$status",
}
path = "$LEDGER"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "a") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
PYEOF
}

run_once() {  # device batch -> exit code
  local device="$1" batch="$2"
  local extra=""
  [ "$FP16" = "1" ] && extra="$extra --fp16"
  [ "$SAVE_MODEL" = "1" ] && extra="$extra --save_model"
  [ "$GRAD_CKPT" = "1" ] && extra="$extra --grad_checkpoint"
  [ "$FREEZE" = "1" ] && extra="$extra --freeze_encoder"
  [ "$ENC_LR_SCALE" != "1.0" ] && extra="$extra --encoder_lr_scale $ENC_LR_SCALE"
  echo "=== [$(date '+%F %T')] task=$TASK device=$device batch=$batch ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  "$PYTHON" -m rnajepa.finetune \
      --task "$TASK" --task_kind "$TASK_KIND" \
      --model_path "$MODEL_PATH" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
      --seed "$SEED" --max_len "$MAX_LEN" --batch_size "$batch" \
      --eval_batch_size "$EVAL_BATCH" --lr "$LR" --epochs "$EPOCHS" \
      --device "$device" $extra >>"$LOG" 2>&1
}

attempt=0
tried=""
batch="$BATCH"
while [ "$attempt" -lt 6 ]; do
  attempt=$((attempt+1))
  device=$(pick_gpu_excluding "$MIN_FREE" "$tried")
  if [ -z "$device" ]; then
    echo "[submit] no GPU with >=${MIN_FREE}MiB free after excluding: $tried" | tee -a "$LOG"
    if [ "$batch" -gt 2 ]; then
      batch=$((batch/2)); tried=""
      echo "[submit] protocol deviation: halving batch to $batch and retrying" | tee -a "$LOG"
      continue
    fi
    ledger_row "no_gpu" "" "$attempt" "$batch"
    exit 3
  fi
  tried="$tried $device"
  ledger_row "running" "$device" "$attempt" "$batch"
  run_once "$device" "$batch"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    ledger_row "success" "$device" "$attempt" "$batch"
    echo "[submit] SUCCESS task=$TASK device=$device batch=$batch" | tee -a "$LOG"
    exit 0
  fi
  if grep -q "CUDA out of memory" "$LOG"; then
    ledger_row "oom" "$device" "$attempt" "$batch"
    echo "[submit] OOM on device=$device; trying another card (batch stays $batch)" | tee -a "$LOG"
    continue
  fi
  ledger_row "failed" "$device" "$attempt" "$batch"
  echo "[submit] FAILED task=$TASK device=$device rc=$rc (not OOM); see $LOG" | tee -a "$LOG"
  exit "$rc"
done

ledger_row "exhausted" "" "$attempt" "$batch"
echo "[submit] EXHAUSTED retries for task=$TASK" | tee -a "$LOG"
exit 4