#!/bin/bash
# Step-4000 replication of the objective-function 2x2, at the *same* decode weight.
#
# Why a second point
# ------------------
# `run_objective_2x2_eval.sh` measures {sum, length} x {aux on, off} at step 2000 with
# w fixed at 1.0 on TS0.  That answers H6 ("are the distillation / calibration terms
# doing anything, or are they decorative?") at one point in time.  One point cannot
# distinguish "the auxiliary terms help" from "the two arms happened to differ at step
# 2000".  This queue repeats the comparison at step 4000, with the decode weight again
# fixed and no selection on any test split, so the answer becomes a trend.
#
# `rinalmo_ff_b4_s0` (sum + aux on) already has a step-2000 TS0 point at w=1.0
# (micro F1 0.4861).  All arms here use `--prior-weight -1` -- the model's own trained
# weight -- so that the same convention is used as in the headline run
# (`astrained_ff3500`) and the step-trend queue.  Using -1 rather than 1.0 is a
# deliberate change from the step-2000 script: -1 involves no constant chosen by us.
#
# Serialisation: waits until no evaluator and none of the other three queue scripts is
# alive, so it cannot race them.  The node is shared (96 cores, load ~115 driven by
# other users' GROMACS jobs) and the evaluator is effectively single-core, so
# concurrency would slow everyone rather than help.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/objective_2x2_step4000.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
STEP="${RNAJEV_2X2B_STEP:-4000}"
W="${RNAJEV_2X2B_W:--1}"

wait_for_others() {
    local waited=0
    while [ "$waited" -lt 1080 ]; do
        if pgrep -f '[e]valuate_decision.py' > /dev/null \
        || pgrep -f '[r]un_followup_evals.sh' > /dev/null \
        || pgrep -f '[r]un_objective_2x2_eval.sh' > /dev/null \
        || pgrep -f '[r]un_step_trend_eval.sh' > /dev/null; then
            sleep 30
            waited=$((waited + 1))
            continue
        fi
        return 0
    done
    return 1
}

{
  echo "waiting for other evaluators and queues $(date '+%T')"
  if ! wait_for_others; then
      echo "FATAL: still busy after 9 h; aborting rather than racing"
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
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$ckpt" \
          --data "$ART/ss_data/jsonl/bprna_ts0.jsonl" \
          --calib-data "$DEV" \
          --prior-weight "$W" \
          --out "$ART/eval_decision/obj2x2b_${tag}_step${STEP}_ts0" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "obj2x2b_${tag}_step${STEP}"
      echo "=== $tag done rc=$? $(date '+%T') ==="
  done
  echo "########## 2x2 step-${STEP} replication done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
