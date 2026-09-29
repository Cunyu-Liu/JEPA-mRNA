#!/bin/bash
# Plan-A convergence extension: continue plana_giga_s0 from its step-20000
# checkpoint to 40000 steps, single variable = training duration.
#
# Rationale (2026-09-29, 6th-round handover): the ff family gained +0.031 TS0
# from 20k -> 40k (TR1 arm, ledger 14.49), and plana's fully-unfrozen phase
# only had 5,600 steps (unfreeze completes at 14,400 of 20,000). Whether the
# 0.7268 headline is converged at 20k is an open question the paper currently
# cannot answer; user directive: train to convergence, do not cap steps.
# Everything else mirrors the s0 protocol exactly (same seed, same LRs, same
# unfreeze schedule -- already complete, resumed state carries it).
#
# GPU policy: needs ~14.1GB peak (s0 measurement). Candidates ordered with
# GPU-5 first (18.6GB free at launch time); any other >=16GB device works.
# RAM: the in-loop trainer's RSS is ~3.4GB (measured on plana_giga_s1), far
# below the 40GB available, so the >60GB ff-era RAM gate does not apply.
set -u

# single-instance guard
exec 9>/mnt/cunyuliu/rna-jepa/runs/.launch_plan_a_ext40k.lock
flock -n 9 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa

SRC=$D/runs/plana_giga_s0
OUT=$D/runs/plana_giga_s0_ext40k
LOG=$D/runs/plana_giga_s0_ext40k.log
STEPS=40000

# idempotency: nothing to do if the extension already finished
if [ -f $D/eval_decision/plana_giga_s0_ext40k_step40000/result.json ]; then
  echo "[ext40k] SKIP: already evaluated"; exit 0
fi

# one-time setup: fresh run dir seeded with s0's step-20000 resume state
mkdir -p "$OUT"
if [ ! -f "$OUT/resume.pt" ]; then
  if [ ! -f "$SRC/resume.pt" ]; then
    echo "[ext40k] FATAL: no resume.pt under $SRC"; exit 1
  fi
  echo "[ext40k] seeding resume.pt from s0 (7.3GB copy) $(date '+%T')"
  cp "$SRC/resume.pt" "$OUT/resume.pt" || { echo "[ext40k] FATAL: copy failed"; exit 1; }
fi

CANDIDATES=(
  "GPU-5"
  "GPU-2"
  "GPU-0"
  "GPU-3"
  "GPU-4"
  "GPU-1"
  "MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
  "MIG-10b9b777-a776-56de-9f7c-efde6c584f71"
)
NEED_GB=16
POLL=120
MAX_WAIT_HOURS=96

free_gb() {
  CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null
}

echo "[ext40k] waiting for a device with >= ${NEED_GB}GB free among: ${CANDIDATES[*]}"

waited=0
DEV=""
while true; do
  for c in "${CANDIDATES[@]}"; do
    fb=$(free_gb "$c")
    if [ -n "$fb" ]; then
      ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
      if [ "$ok" = "1" ]; then
        DEV="$c"; break 2
      fi
    fi
  done
  if [ "$waited" -ge $((MAX_WAIT_HOURS * 3600 / POLL)) ]; then
    echo "[ext40k] FATAL: no device freed up after ${MAX_WAIT_HOURS}h"; exit 1
  fi
  sleep "$POLL"
  waited=$((waited + 1))
done

echo "[ext40k] acquired $DEV (free ${fb}GB); launching $(date '+%T')"

cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
  --out "$OUT" \
  --data "$D/ss_data/jsonl/bprna_tr0.jsonl" \
  --teacher-dir "$D/ss_data/teacher/bprna_tr0" \
  --steps "$STEPS" \
  --batch-size 4 \
  --head-lr 1e-4 --backbone-lr 1e-5 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --save-every 500 --snapshot-every 2000 --seed 0 --gpu-reserve-gb 14 \
  9>&- > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$OUT.launch_pid"
echo "$DEV" > "$OUT.launch_dev"
echo "[ext40k] launched pid=$PID on $DEV; log -> $LOG"
