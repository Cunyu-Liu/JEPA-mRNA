#!/bin/bash
# The 2x2 objective comparison at a *matched step*, on TS0.
#
# `rinalmo_ff_b4_s0` (sum + aux on) already has a step-2000 TS0 result at w=1.0:
# micro F1 0.4861, raw ECE 0.2726, C1-c gap 0.2631.  Evaluating the three new arms at
# the same step and the same decode weight makes the comparison a single-variable one,
# which is the only way the H6 question ("is RLCD / distillation doing anything, or are
# they decorative?") can be answered at all.
#
#   sum    + aux off : rinalmo_sum_b4_s0
#   sum    + aux on  : rinalmo_ff_b4_s0   (already measured)
#   length + aux off : rinalmo_len_b4_s0
#   length + aux on  : rinalmo_bal_b4_s0
#
# Decode weight is fixed at 1.0 (not VL0-selected) for every arm, so no selection
# enters the comparison.  The calibration map is fitted on bpRNA VL0, which is
# disjoint from TS0, as the protocol requires.
#
# Waits for any running evaluator rather than racing it: the node is shared and the
# O(L^3) DP is the bottleneck.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/objective_2x2_step2000.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
STEP="${RNAJEV_2X2_STEP:-2000}"
W="${RNAJEV_2X2_W:-1.0}"

{
  echo "waiting for other evaluators $(date '+%T')"
  for _ in $(seq 1 720); do
      pgrep -f '[e]valuate_decision.py' > /dev/null || break
      sleep 30
  done
  if pgrep -f '[e]valuate_decision.py' > /dev/null; then
      echo "FATAL: another evaluator is still running after 6 h"
      exit 1
  fi
  echo "clear to run $(date '+%T')"

  for tag in rinalmo_sum_b4_s0 rinalmo_len_b4_s0 rinalmo_bal_b4_s0; do
      ckpt="$ART/ckpts/${tag}_step${STEP}.pt"
      if [ ! -f "$ckpt" ]; then
          echo "=== SKIP $tag: no snapshot at $ckpt ==="
          continue
      fi
      echo "=== $tag step=$STEP w=$W on bprna_ts0 $(date '+%T') ==="
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 10 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$ckpt" \
          --data "$ART/ss_data/jsonl/bprna_ts0.jsonl" \
          --calib-data "$DEV" \
          --prior-weight "$W" \
          --out "$ART/eval_decision/obj2x2_${tag}_step${STEP}_ts0" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "obj2x2_${tag}_step${STEP}"
      echo "=== $tag done rc=$? $(date '+%T') ==="
  done
  echo "########## 2x2 comparison done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
