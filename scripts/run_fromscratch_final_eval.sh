#!/bin/bash
# Final evals for the from-scratch (own-encoder) arms that reached step 40000.
# Answers: does a from-scratch 35M encoder close any gap to the frozen RiNALMo
# embeddings at the SAME 20k steps, and where does it end at 40k?
# from-scratch models have no embedding store -> no --embedding-dir.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
TEST=$ART/ss_data/jsonl/bprna_ts0.jsonl
LOG=$ART/eval_decision/fromscratch_final_eval.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_s2_eval_watch.sh" >/dev/null && { sleep 30; continue; }
      break
  done
}

{
  for spec in "full_b4_s0_20260924T060638:40000" "full_b4_s0_20260924T060638:20000" "nlldistill_b4_s0_20260924T060638:40000" "nlldistill_b4_s0_20260924T060638:20000"; do
      arm=$(echo "$spec" | cut -d: -f1)
      step=$(echo "$spec" | cut -d: -f2)
      ckpt="$ART/ckpts/${arm}_step${step}.pt"
      [ -f "$ckpt" ] || { echo "SKIP $arm $step: no snapshot"; continue; }
      wait_quiet
      echo "=== $arm step=$step w=-1 ts0 ==="
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$ckpt" \
          --data "$TEST" \
          --calib-data "$DEV" \
          --prior-weight -1 \
          --out "$ART/eval_decision/fs_${arm}_step${step}_ts0" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "fs_${arm}_step${step}_ts0"
      echo "=== $arm $step done rc=$? ==="
  done
  echo "########## from-scratch final eval done ##########"
} >> "$LOG" 2>&1
echo detached
