#!/bin/bash
# Step-trend evaluation for the headline arm, all at the *same* decode weight.
#
# Why this exists
# ---------------
# The report needs an F1-vs-steps curve for `rinalmo_ff`, not a single point.  Every
# existing TS0 measurement of that arm used a *different* decode weight, so the points
# are not comparable:
#
#   step 1000 -> 0.4554 (w=0.5)   step 2000 -> 0.4861 (w=1.0)   step 3500 -> 0.4959 (w=0.75)
#
# Here every point is measured with `--prior-weight -1`, i.e. the model's own trained
# prior weight with no selection on any split.  That is the same protocol as the
# in-flight `astrained_ff3500` headline run, so the curve and the headline are one
# consistent measurement.  Step numbers are read from inside the checkpoints, not from
# file names (see records/DECISION_TRAINING_LOG.md T-A8).
#
# Serialisation: this waits until *no* other evaluator and *no* other eval queue script
# is alive, so it cannot race the follow-up queue or the 2x2 queue that are already
# waiting.  The node is shared (96 cores, load ~110) and the O(L^3) numpy DP is the
# bottleneck, so concurrency would slow everyone down rather than help.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/step_trend.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

STEPS="${RNAJEV_TREND_STEPS:-6000 10000}"
W="${RNAJEV_TREND_W:--1}"

{
  echo "waiting for other evaluators and queues $(date '+%T')"
  for _ in $(seq 1 960); do
      pgrep -f '[e]valuate_decision.py' > /dev/null && { sleep 30; continue; }
      pgrep -f '[r]un_followup_evals.sh' > /dev/null && { sleep 30; continue; }
      pgrep -f '[r]un_objective_2x2_eval.sh' > /dev/null && { sleep 30; continue; }
      break
  done
  if pgrep -f '[e]valuate_decision.py' > /dev/null; then
      echo "FATAL: still busy after 8 h; aborting rather than racing"
      exit 1
  fi
  echo "clear to run $(date '+%T')"

  for step in $STEPS; do
      ckpt="$ART/ckpts/rinalmo_ff_b4_s0_step${step}.pt"
      if [ ! -f "$ckpt" ]; then
          echo "=== SKIP step=$step: no snapshot at $ckpt ==="
          continue
      fi
      echo "=== rinalmo_ff step=$step w=$W on bprna_ts0 $(date '+%T') ==="
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$ckpt" \
          --data "$ART/ss_data/jsonl/bprna_ts0.jsonl" \
          --calib-data "$DEV" \
          --prior-weight "$W" \
          --out "$ART/eval_decision/trend_ff_step${step}_ts0" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "trend_ff_step${step}_ts0"
      echo "=== step=$step done rc=$? $(date '+%T') ==="
  done
  echo "########## step trend done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
