#!/bin/bash
# Eval queue: ff seed s1 (seed variance, T-A14) + big capacity arm (T-A15).
# Same serial protocol as run_batch_arms_eval.sh; waits for other queues.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
TEST=$ART/ss_data/jsonl/bprna_ts0.jsonl
LOG=$ART/eval_decision/seeds_capacity_eval.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
: > "$LOG"

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_batch_arms_eval.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_affine_probe.sh" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_followup_evals.sh" >/dev/null && { sleep 30; continue; }
      break
  done
}

{
  for spec in "rinalmo_ff_b4_s1:20000:64" "rinalmo_big_b4_s0:20000:16"; do
      arm=$(echo "$spec" | cut -d: -f1)
      step=$(echo "$spec" | cut -d: -f2)
      chunk=$(echo "$spec" | cut -d: -f3)
      ckpt="$ART/ckpts/${arm}_step${step}.pt"
      [ -f "$ckpt" ] || { echo "SKIP $arm: no snapshot"; continue; }
      wait_quiet
      echo "=== $arm step=$step w=-1 ts0 chunk=$chunk ==="
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
      echo "=== $arm done rc=$? ==="
  done
  echo "########## seeds/capacity eval done ##########"
} >> "$LOG" 2>&1
echo "detached"
