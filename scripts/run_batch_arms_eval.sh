#!/bin/bash
# Batch eval queue for arms that reached step 20000 on 2026-09-24 evening.
# Serial by design: the O(L^3) numpy DP is the bottleneck on this shared node
# (96 cores, load ~110), so concurrency would slow everyone down.
# Protocol matches the headline: --prior-weight -1 (as-trained, no selection),
# --calib-data VL0 for the DP-free affine recalibration, TS0 as the test split.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
TEST=$ART/ss_data/jsonl/bprna_ts0.jsonl
LOG=$ART/eval_decision/batch_arms_2200.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
: > "$LOG"

wait_quiet() {
  for _ in $(seq 1 960); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_followup_evals.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_objective_2x2_eval.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_step_trend_eval.sh" >/dev/null && { sleep 30; continue; }
      break
  done
}

{
  echo "queue start $(date \x27+%T\x27)"
  for spec in \
      "rinalmo_len_b4_s0:20000" \
      "rinalmo_sum_b4_s0:20000" \
      "rinalmo_pw_b4_s0:20000" \
      "rinalmo_bal_b4_s0:20000" \
      "rinalmo_bal_b4_s1:20000" \
      ; do
      arm="${spec%%:*}"; step="${spec##*:}"
      ckpt="$ART/ckpts/${arm}_step${step}.pt"
      if [ ! -f "$ckpt" ]; then
          ckpt_alt="$ART/runs/${arm}/resume.pt"
          [ -f "$ckpt_alt" ] && ckpt="$ckpt_alt" || { echo "SKIP $arm: no ckpt"; continue; }
      fi
      wait_quiet
      echo "=== $arm step=$step w=-1 ts0 $(date \x27+%T\x27) ==="
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$ckpt" \
          --data "$TEST" \
          --calib-data "$DEV" \
          --prior-weight -1 \
          --out "$ART/eval_decision/arms_${arm}_step${step}_ts0" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "arms_${arm}_step${step}_ts0"
      echo "=== $arm done rc=$? $(date \x27+%T\x27) ==="
  done
  echo "########## batch arms done $(date \x27+%T\x27) ##########"
} >> "$LOG" 2>&1
echo "detached: log -> $LOG"
