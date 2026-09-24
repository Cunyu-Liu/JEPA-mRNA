#!/bin/bash
# The VL0-independent headline numbers, run after the VL0 sweep job finishes.
#
# Why: §14.15 problem 2.  On the *same* length bucket (<=100 nt) the same checkpoint
# scores micro F1 0.9230 on bpRNA VL0 but 0.6188 on bpRNA TS0 -- a 0.30 gap with no
# identified cause.  Exact sequence overlap is zero, K=20 containment against TR0 is
# 0.042 vs 0.040, and length / pair-density / family-label composition are matched.
# Since VL0 cannot be trusted as a proxy for TS0, the headline number must not depend
# on a weight selected there.
#
# So this run reports `--prior-weight -1`, which means "use the model's own trained
# prior weight" -- no selection on any split at all.  The VL0-selected w is kept as a
# sensitivity analysis, not as the headline.
#
# It waits for the running evaluate_decision.py to disappear rather than racing it:
# the node is shared and the O(L^3) DP is the bottleneck, so two evaluators at once
# would just halve both.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
CKPT=$ART/ckpts/rinalmo_ff_ff_w05_snapshot.pt
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/ood_astrained.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

{
  echo "waiting for the VL0-sweep evaluator to finish $(date '+%T')"
  for _ in $(seq 1 720); do
      if ! pgrep -f '[e]valuate_decision.py' > /dev/null; then break; fi
      sleep 30
  done
  if pgrep -f '[e]valuate_decision.py' > /dev/null; then
      echo "FATAL: another evaluator is still running after 6 h; aborting rather than racing it"
      exit 1
  fi
  echo "clear to run $(date '+%T')"

  for split in bprna_ts0 archiveii bprna_new; do
      echo "=== as-trained (w=-1) on $split $(date '+%T') ==="
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 10 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$CKPT" \
          --data "$ART/ss_data/jsonl/$split.jsonl" \
          --calib-data "$DEV" \
          --prior-weight -1 \
          --out "$ART/eval_decision/astrained_ff3500_$split" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "astrained_ff3500_$split"
      echo "=== as-trained $split done rc=$? $(date '+%T') ==="
  done
  echo "########## as-trained evals done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
