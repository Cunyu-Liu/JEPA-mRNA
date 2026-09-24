#!/bin/bash
# Evaluate the architecture arms that have never been scored on F1.
#
# Why these three
# ---------------
#   rinalmo_casc_b4_s0   (miss_cost=20,  sparse=0.1)  -- C2, the hierarchical cascade
#   rinalmo_cascR_b4_s0  (miss_cost=200, sparse=0.02) -- C2, the recall-preserving variant
#   rinalmo_big_b4_s0    (d_z=512, hidden=512, 4.8x params) -- the head-capacity arm
#
# C2 is one of the two claimed contributions and its F1 has never been measured.  All
# that exists is the training-time L0 helix recall, which is a *training-set* quantity
# and cannot support the claim (spec: no training-set metric may be written up as a
# scientific result).  The two cascade arms bracket the density-recall trade-off:
# `casc` was drifting below the P6 recall gate (0.88 at step 1350) while `cascR` held
# it (0.996), so evaluating both tells us whether the gate actually matters for F1.
#
# The capacity arm answers a separate question left open since 14.19: is the flat
# head's plateau a capacity limit?  It has 2,507,791 trainable parameters against the
# flat head's 519,503, with everything else identical.
#
# All three are scored at step 2000 with `--prior-weight -1` (the model's own trained
# weight) on TS0, so they are directly comparable to each other and to the headline
# convention.  The evaluator detects a cascade checkpoint by `hasattr(head,
# "forward_gated")` and routes it through `forward_gated`, so no extra flag is needed;
# passing the flat path would score a head that was never trained.
#
# Serialisation: waits for every other evaluator and queue script, since the node is
# shared and the evaluator is effectively single-core.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/eval
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
JSONL=$ART/ss_data/jsonl
DEV=$JSONL/bprna_vl0.jsonl
LOG=$ART/eval_decision/arch_arms.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
STEP="${RNAJEV_ARCH_STEP:-2000}"
W="${RNAJEV_ARCH_W:--1}"

wait_for_others() {
    local waited=0
    while [ "$waited" -lt 1200 ]; do
        if pgrep -f '[e]valuate_decision.py' > /dev/null \
        || pgrep -f '[r]un_followup_evals.sh' > /dev/null \
        || pgrep -f '[r]un_objective_2x2_eval.sh' > /dev/null \
        || pgrep -f '[r]un_step_trend_eval.sh' > /dev/null \
        || pgrep -f '[r]un_objective_2x2_step4000_eval.sh' > /dev/null \
        || pgrep -f '[r]un_clean_split_evals.sh' > /dev/null \
        || pgrep -f '[l]ength_bucketed_leaderboard.py' > /dev/null; then
            sleep 30
            waited=$((waited + 1))
            continue
        fi
        return 0
    done
    return 1
}

{
  echo "waiting for the other queues $(date '+%T')"
  if ! wait_for_others; then
      echo "FATAL: still busy after 10 h; aborting rather than racing"
      exit 1
  fi
  echo "clear to run $(date '+%T')"

  for tag in rinalmo_casc_b4_s0 rinalmo_cascR_b4_s0 rinalmo_big_b4_s0; do
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
          --data "$JSONL/bprna_ts0.jsonl" \
          --calib-data "$DEV" \
          --prior-weight "$W" \
          --out "$ART/eval_decision/arch_${tag}_step${STEP}_ts0" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "arch_${tag}_step${STEP}"
      echo "=== $tag done rc=$? $(date '+%T') ==="
  done
  echo "########## architecture arms done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
