#!/bin/bash
# Submit one decision-model pretraining / finetuning job (Task 18).
#
# Thin launcher around `python -m rnajepa.train_decision`.  It follows the same
# conventions as the rest of scripts/: it picks the emptiest GPU itself, is
# idempotent (an arm whose run_meta.json already reports "completed" is skipped),
# supports --dry-run without touching a GPU, and appends a ledger row so the
# experiment bookkeeping is never reconstructed from memory.
#
# Learning-rate calibration is NOT inherited from any default: the earlier phase of
# this project measured that the documented lr=1e-4 destroys the pretrained encoder
# on tasks with >10k rows (records/R3_GATE_AND_LR_CALIBRATION.md).  Pass --lr-calibrate
# (the default here) and the driver records the measured value in run_meta.json.
#
# Usage:
#   RNAJEPA_PYTHON=<py> run_pretrain.sh --arm main --out <run_dir> \
#       [--data <jsonl>] [--teacher-dir <dir>] [--steps N] [--batch-size B] \
#       [--lr LR | --lr-calibrate] [--lambda-nll X --lambda-distill X \
#        --lambda-rlcd X --lambda-cal X] [--distill-kind kl|l2] \
#       [--rlcd-reward brier|log] [--beta X] [--device N] [--min-free MIB] \
#       [--dry-run] [--force]
#
# Dry-run (no GPU needed, no training):
#   RNAJEPA_PYTHON=python run_pretrain.sh --arm smoke --dry-run --synthetic 8 --tiny
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
PY="${RNAJEPA_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"

ARM="main"; OUT=""; DATA=""; TEACHER_DIR=""; SYNTHETIC=0; LENGTH=24
STEPS=20000; BATCH=8; LR=""; WEIGHT_DECAY=0.0; WARMUP=1000; GRAD_CLIP=1.0
LAMBDA_NLL=1.0; LAMBDA_DISTILL=1.0; LAMBDA_RLCD=1.0; LAMBDA_CAL=1.0
DISTILL_KIND="kl"; RLCD_REWARD="brier"; BETA=1.0
LR_CALIBRATE=1; LR_CANDIDATES="1e-4,5e-5,1e-5"; LR_PROBE_STEPS=50
DEVICE=""; MIN_FREE=14000; MAX_HOURS=20; SAVE_EVERY=1000; LOG_EVERY=20
SEED=42; TINY=0; DRY=0; FORCE=0; RESUME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --teacher-dir) TEACHER_DIR="$2"; shift 2;;
    --synthetic) SYNTHETIC="$2"; shift 2;;
    --length) LENGTH="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --lr) LR="$2"; LR_CALIBRATE=0; shift 2;;
    --lr-calibrate) LR_CALIBRATE=1; shift;;
    --lr-candidates) LR_CANDIDATES="$2"; shift 2;;
    --lr-probe-steps) LR_PROBE_STEPS="$2"; shift 2;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2;;
    --warmup-steps) WARMUP="$2"; shift 2;;
    --grad-clip) GRAD_CLIP="$2"; shift 2;;
    --lambda-nll) LAMBDA_NLL="$2"; shift 2;;
    --lambda-distill) LAMBDA_DISTILL="$2"; shift 2;;
    --lambda-rlcd) LAMBDA_RLCD="$2"; shift 2;;
    --lambda-cal) LAMBDA_CAL="$2"; shift 2;;
    --distill-kind) DISTILL_KIND="$2"; shift 2;;
    --rlcd-reward) RLCD_REWARD="$2"; shift 2;;
    --beta) BETA="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --min-free) MIN_FREE="$2"; shift 2;;
    --max-hours) MAX_HOURS="$2"; shift 2;;
    --save-every) SAVE_EVERY="$2"; shift 2;;
    --log-every) LOG_EVERY="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --tiny) TINY=1; shift;;
    --resume) RESUME="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --force) FORCE=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

[ -z "$OUT" ] && { echo "missing required --out" >&2; exit 2; }
if [ "$SYNTHETIC" = "0" ] && [ -z "$DATA" ]; then
  echo "give --data <jsonl> or --synthetic N" >&2; exit 2
fi
if [ -n "$DATA" ] && [ "$LAMBDA_NLL" != "0" ] && [ "$LAMBDA_NLL" != "0.0" ] \
   && [ ! -f "$DATA" ]; then
  echo "FATAL: --data $DATA does not exist (hard labels are required for L_NLL)" >&2
  exit 2
fi

LEDGER="${RNAJEPA_LEDGER:-$ART/ledger.jsonl}"
export PYTHONPATH="${RNAJEPA_PYEXT:-/mnt/cunyuliu/rna-jepa/pyext}:$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date '+%F %T')] $*"; }

# ------------------------------------------------------------------ args
ARGS=( -m rnajepa.train_decision --arm "$ARM" --out "$OUT" --steps "$STEPS"
       --batch-size "$BATCH" --weight-decay "$WEIGHT_DECAY"
       --warmup-steps "$WARMUP" --grad-clip "$GRAD_CLIP" --seed "$SEED"
       --lambda-nll "$LAMBDA_NLL" --lambda-distill "$LAMBDA_DISTILL"
       --lambda-rlcd "$LAMBDA_RLCD" --lambda-cal "$LAMBDA_CAL"
       --distill-kind "$DISTILL_KIND" --rlcd-reward "$RLCD_REWARD" --beta "$BETA"
       --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY"
       --max-hours "$MAX_HOURS" )
if [ "$SYNTHETIC" != "0" ]; then ARGS+=( --synthetic "$SYNTHETIC" --length "$LENGTH" ); fi
if [ -n "$DATA" ]; then ARGS+=( --data "$DATA" ); fi
if [ -n "$TEACHER_DIR" ]; then ARGS+=( --teacher-dir "$TEACHER_DIR" ); fi
if [ -n "$RESUME" ]; then ARGS+=( --resume "$RESUME" ); fi
if [ "$TINY" = "1" ]; then ARGS+=( --tiny ); fi
if [ "$LR_CALIBRATE" = "1" ]; then
  ARGS+=( --lr-calibrate --lr-candidates "$LR_CANDIDATES" --lr-probe-steps "$LR_PROBE_STEPS" )
else
  ARGS+=( --lr "$LR" )
fi

# ------------------------------------------------------------------ dry run
if [ "$DRY" = "1" ]; then
  log "[dry-run] arm=$ARM steps=$STEPS batch=$BATCH out=$OUT"
  [ "$LR_CALIBRATE" = "1" ] && log "[dry-run] lr calibration over {$LR_CANDIDATES}" \
                            || log "[dry-run] lr=$LR (calibration disabled)"
  "$PY" "${ARGS[@]}" --dry-run
  exit $?
fi

# ------------------------------------------------------------------ idempotency
if [ "$FORCE" = "0" ] && [ -f "$OUT/run_meta.json" ]; then
  if grep -q '"status": "completed"' "$OUT/run_meta.json" 2>/dev/null; then
    log "SKIP $ARM: $OUT/run_meta.json reports a completed run (use --force to redo)"
    exit 0
  fi
fi

# ------------------------------------------------------------------ GPU pick
if [ -n "$DEVICE" ]; then
  GPU="$DEVICE"
else
  source "$ROOT/scripts/gpu_util.sh"
  GPU=$(pick_gpu "$MIN_FREE") || { log "no GPU with >=${MIN_FREE}MiB free"; exit 3; }
fi
ARGS+=( --device "$GPU" )

mkdir -p "$OUT"
log "launching arm=$ARM device=$GPU -> $OUT"
nohup "$PY" "${ARGS[@]}" > "$OUT.stdout.log" 2>&1 &
PID=$!
log "pid $PID writing $OUT.stdout.log"

python3 - "$LEDGER" "$ARM" "$OUT" "$GPU" "$PID" <<'PYEOF'
import json, os, socket, sys, datetime
ledger, arm, out, gpu, pid = sys.argv[1:6]
os.makedirs(os.path.dirname(ledger), exist_ok=True)
row = {"ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
       "host": socket.gethostname(), "stage": "decision_pretrain", "arm": arm,
       "out_dir": out, "device": gpu, "pid": int(pid), "status": "dispatched"}
with open(ledger, "a") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
PYEOF

log "dispatched; follow with: tail -f $OUT.stdout.log"
