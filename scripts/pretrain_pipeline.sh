#!/bin/bash
# Unattended pre-training pipeline: wait for the corpus, verify it, preprocess it,
# then launch the smoke stage followed by the two full pre-training arms.
#
# Runs inside tmux so a dropped SSH session cannot interrupt it.  Every stage is
# idempotent: an existing verified artefact is reused rather than recomputed, and
# the corpus md5 is checked before a single sequence is processed.
#
# Usage: RNAJEPA_PYTHON=<python> pretrain_pipeline.sh [--skip-smoke] [--workers N]
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
PY="${RNAJEPA_PYTHON:-/home/cunyuliu/miniconda3/envs/mrnabert/bin/python}"
[ -x "$PY" ] || PY="/home/cunyuliu/miniconda3/envs/lucaone/bin/python"
WORKERS=96
SKIP_SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --skip-smoke) SKIP_SMOKE=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

ZIP="$ART/data/raw/mRNAdataset.zip"
ZIP_MD5="bf8bc5c946a0bd3b07716b1c7f785d54"
PRE="$ART/data/pretrain/pre.txt"
PRE_REG="$ART/data/pretrain/pre_regions.txt"
STATS="$ART/data/pretrain/stats.json"
WEIGHTS="$ART/weights/mRNABERT_attn_fallback"
LOG="$ART/logs/pretrain_pipeline.log"

export PYTHONPATH="${RNAJEPA_PYEXT:-/mnt/cunyuliu/rna-jepa/pyext}:$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
source "$ROOT/scripts/gpu_util.sh"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. corpus
say "waiting for $ZIP"
while [ ! -f "$ZIP" ]; do sleep 60; done
for _ in $(seq 1 30); do
  got=$(md5sum "$ZIP" | cut -d' ' -f1)
  if [ "$got" = "$ZIP_MD5" ]; then break; fi
  say "md5 not final yet ($got); rechecking in 60s"
  sleep 60
done
if [ "$(md5sum "$ZIP" | cut -d' ' -f1)" != "$ZIP_MD5" ]; then
  say "FATAL: corpus md5 mismatch; refusing to preprocess a corrupt archive"
  exit 1
fi
say "corpus verified ($(du -h "$ZIP" | cut -f1))"

# ---------------------------------------------------------------- 2. preprocess
if [ -f "$STATS" ] && [ -s "$PRE_REG" ]; then
  say "preprocessed corpus already present; skipping"
else
  say "preprocessing with $WORKERS workers"
  "$PY" "$ROOT/data/prep_pretrain.py" --zip "$ZIP" \
      --work "$ART/data/pretrain_work" --out "$PRE" --out_regions "$PRE_REG" \
      --stats "$STATS" --workers "$WORKERS" --verify 20000 2>&1 | tee -a "$LOG"
  [ -f "$STATS" ] || { say "FATAL: preprocessing produced no stats.json"; exit 1; }
fi
N_SEQ=$("$PY" -c "import json;print(json.load(open('$STATS'))['n'])" 2>/dev/null || echo 0)
say "corpus sequences: $N_SEQ"
# Step budget rationale: the run must be evaluated *while* it trains, because the node is
# shared with ~70 other users and wall-clock is unpredictable.  Checkpoints every 5000
# steps (12 of them at 40k) let the learning curve be probed as it goes and the best
# checkpoint taken, instead of waiting for a fixed final step.  --max_hours guarantees a
# clean save-and-exit rather than an indefinite stall; resume.pt continues it later.
# max_len 1024 with batch 4 x accum 8 keeps the effective batch at 32 sequences per
# optimiser step while giving each region a chance to appear, and the reader crops from
# the head or the tail deterministically so the 3'UTR is actually seen.
say "pretrain budget: micro-steps=${PRETRAIN_STEPS:-80000} (10k opt steps) save_every=${SAVE_EVERY:-5000} max_hours=${MAX_HOURS:-20} max_len=1024 batch=4x8"

# ---------------------------------------------------------------- 3. smoke
if [ "$SKIP_SMOKE" = "0" ]; then
  for A in 0.25 0.5 1.0; do
    name="smoke_alpha${A}"
    out="$ART/runs/${name}"
    if [ -f "$out/train_log.jsonl" ]; then say "$name already done"; continue; fi
    G=$(pick_gpu 12000) || { say "no GPU for $name"; continue; }
    say "smoke alpha=$A on gpu$G"
    "$PY" -m rnajepa.pretrain --arm "$name" --init official \
        --data "$PRE" --regions "$PRE_REG" --weights "$WEIGHTS" --out "$out" \
        --device "$G" --batch_size 4 --grad_accum 8 --steps 800 --warmup_steps 50 \
        --lr 5e-5 --max_len 1024 --save_every 0 --log_every 20 \
        --alpha "$A" --jepa_target both 2>&1 | tee -a "$LOG" | grep -E "^\[|DONE"
  done
fi

# ---------------------------------------------------------------- 4. full arms
# V1 = continual from the released weights (main result)
# V2 = from scratch (fair-comparison arm)
for ARM in v1_cont v2_scratch; do
  if [ "$ARM" = "v1_cont" ]; then INIT=official; else INIT=random; fi
  out="$ART/runs/${ARM}"
  if [ -f "$out/train_log.jsonl" ] && [ -z "${REDO:-}" ]; then
    # only skip if a checkpoint exists too
    if ls "$out"/hf/step_* >/dev/null 2>&1; then say "$ARM already has checkpoints; skipping"; continue; fi
  fi
  G=$(pick_gpu 14000) || { say "no GPU for $ARM"; continue; }
  say "launching $ARM (init=$INIT) on gpu$G"
  nohup "$PY" -m rnajepa.pretrain --arm "$ARM" --init "$INIT" \
      --data "$PRE" --regions "$PRE_REG" --weights "$WEIGHTS" --out "$out" \
      --device "$G" --batch_size 4 --grad_accum 8 --steps "${PRETRAIN_STEPS:-80000}" \
      --warmup_steps 1000 \
      --lr 5e-5 --max_len 1024 --save_every "${SAVE_EVERY:-5000}" --log_every 50 \
      --alpha 0.5 --jepa_target both --max_hours "${MAX_HOURS:-20}" \
      > "$out.stdout.log" 2>&1 &
  say "$ARM pid $! writing $out.stdout.log"
done

say "pipeline finished dispatching; arms run in the background"