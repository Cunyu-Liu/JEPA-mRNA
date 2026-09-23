#!/bin/bash
# Launch one decision-model training run on the emptiest schedulable GPU.
#
# This is the structure task's counterpart to submit_finetune.sh.  It exists
# because there is no job scheduler on this node and because the node is shared:
# GPUs 0-5 are usually busy with other people's work, while GPU 6 (7 x 1g.5gb)
# and GPU 7 (2 x 3g.20gb) are MIG-partitioned and often idle.  gpu_util.sh
# enumerates MIG instances by UUID, which is the only unambiguous way to select
# one; setting the parent index leaves it undefined which instance you get.
#
# Discipline inherited from the rest of this repo:
#   * CUDA_VISIBLE_DEVICES is set *before* python starts, so a device-selection
#     mistake cannot be masked by a later torch import;
#   * train_decision.py itself refuses device="cpu" without --allow-cpu, so a
#     silent CPU run is impossible even if this script is bypassed;
#   * an OOM retries on another card rather than halving the batch silently;
#   * every attempt appends a row to the ledger.
#
# Usage:
#   bash scripts/run_decision_training.sh --arm decision --steps 4000 \
#        --data <jsonl> --teacher-dir <dir> --encoder-size 35M [--tag NAME]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
# shellcheck source=scripts/gpu_util.sh
source "$HERE/gpu_util.sh"

PY="${RNAJEV_PYTHON:-$HOME/miniconda3/envs/lucaone/bin/python}"
# ViennaRNA lives out-of-tree: the /home quota is full, so it is installed under
# /mnt/cunyuliu/pylibs and reached through PYTHONPATH.
export PYTHONPATH="${RNAJEV_PYTHONPATH:-/mnt/cunyuliu/pylibs}:$ROOT/src"
export TMPDIR="${RNAJEV_TMPDIR:-/mnt/cunyuliu/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUNS_ROOT="${RNAJEV_RUNS:-/mnt/cunyuliu/rna-jepa/runs}"
LEDGER="$RUNS_ROOT/ledger.jsonl"
mkdir -p "$RUNS_ROOT" "$TMPDIR"

# ---- defaults -------------------------------------------------------------
ARM=decision
DATA=""
TEACHER_DIR=""
STEPS=2000
BATCH=2
LR=1e-4
ENCODER=35M
SEED=0
MIN_FREE="${RNAJEV_MIN_FREE:-4000}"
TAG=""
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --arm)         ARM="$2"; shift 2;;
    --data)        DATA="$2"; shift 2;;
    --teacher-dir) TEACHER_DIR="$2"; shift 2;;
    --steps)       STEPS="$2"; shift 2;;
    --batch)       BATCH="$2"; shift 2;;
    --lr)          LR="$2"; shift 2;;
    --encoder-size) ENCODER="$2"; shift 2;;
    --seed)        SEED="$2"; shift 2;;
    --min-free)    MIN_FREE="$2"; shift 2;;
    --tag)         TAG="$2"; shift 2;;
    *)             EXTRA+=("$1"); shift;;
  esac
done

if [ -z "$DATA" ]; then
  echo "FATAL: --data is required" >&2; exit 2
fi
if [ ! -f "$DATA" ]; then
  echo "FATAL: no such corpus: $DATA" >&2; exit 2
fi
if [ -n "$TEACHER_DIR" ] && [ ! -f "$TEACHER_DIR/manifest.json" ]; then
  echo "FATAL: --teacher-dir has no manifest.json: $TEACHER_DIR" >&2; exit 2
fi

STAMP="$(date +%Y%m%dT%H%M%S)"
NAME="${TAG:-${ARM}}_${ENCODER}_s${SEED}_${STAMP}"
OUT_DIR="$RUNS_ROOT/$NAME"

ledger() {  # ledger <json-fragment>
  printf '%s\n' "{\"ts\":\"$(date -Iseconds)\",\"run\":\"$NAME\",$1}" >> "$LEDGER"
}

echo "[launch] $NAME"
echo "[launch] data=$DATA teacher=${TEACHER_DIR:-<none: hard labels only>}"
echo "[launch] steps=$STEPS batch=$BATCH lr=$LR encoder=$ENCODER seed=$SEED"

TRIED=""
ATTEMPT=0
MAX_ATTEMPTS="${RNAJEV_MAX_ATTEMPTS:-4}"

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  TARGET="$(pick_gpu_excluding "$MIN_FREE" "$TRIED")"
  if [ -z "$TARGET" ]; then
    echo "[launch] no GPU with >= ${MIN_FREE} MiB free; tried: ${TRIED:-none}" >&2
    ledger "\"status\":\"no_capacity\",\"min_free\":$MIN_FREE"
    exit 3
  fi
  TRIED="$TRIED $TARGET"

  echo "[launch] attempt $ATTEMPT -> device $TARGET"
  ledger "\"status\":\"start\",\"device\":\"$TARGET\",\"attempt\":$ATTEMPT,\"steps\":$STEPS,\"encoder\":\"$ENCODER\""

  ARGS=(--arm "$ARM" --out "$OUT_DIR" --data "$DATA" --steps "$STEPS"
        --batch-size "$BATCH" --lr "$LR" --encoder-size "$ENCODER"
        --seed "$SEED" --device cuda --log-every 10)
  if [ -n "$TEACHER_DIR" ]; then
    ARGS+=(--teacher-dir "$TEACHER_DIR")
  fi
  if [ "${#EXTRA[@]}" -gt 0 ]; then
    ARGS+=("${EXTRA[@]}")
  fi

  # device selection happens before python starts
  CUDA_VISIBLE_DEVICES="$TARGET" "$PY" -m rnajepa.train_decision "${ARGS[@]}" \
      2>&1 | tee "$RUNS_ROOT/${NAME}.log"
  STATUS="${PIPESTATUS[0]}"

  if [ "$STATUS" -eq 0 ]; then
    echo "[launch] OK -> $OUT_DIR"
    ledger "\"status\":\"completed\",\"device\":\"$TARGET\",\"attempt\":$ATTEMPT,\"out\":\"$OUT_DIR\""
    echo "$OUT_DIR"
    exit 0
  fi

  if grep -qiE "out of memory|CUDA error: out of memory|CUDA_ERROR_OUT_OF_MEMORY" \
        "$RUNS_ROOT/${NAME}.log"; then
    echo "[launch] OOM on $TARGET; retrying on another device"
    ledger "\"status\":\"oom\",\"device\":\"$TARGET\",\"attempt\":$ATTEMPT"
    continue
  fi

  echo "[launch] FAILED (exit $STATUS) on $TARGET; not retrying (not an OOM)" >&2
  ledger "\"status\":\"failed\",\"device\":\"$TARGET\",\"attempt\":$ATTEMPT,\"exit\":$STATUS"
  exit 1
done

echo "[launch] exhausted $MAX_ATTEMPTS attempts" >&2
ledger "\"status\":\"exhausted\",\"attempts\":$MAX_ATTEMPTS"
exit 1
