#!/bin/bash
# TR1 final evals: the 4.29x-data arms. Waits for the step-20000 snapshot,
# then evaluates TS0 AND bpRNA-new (the cross-family split) with w=-1.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/tr1_final_eval.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
CKPT="$ART/ckpts/rinalmo_ff_tr1_b4_s0_step20000.pt"

wait_quiet() {
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_fromscratch_final_eval.sh" >/dev/null && { sleep 30; continue; }
      break
  done
}

{
  for _ in $(seq 1 2400); do
      [ -f "$CKPT" ] && break
      sleep 60
  done
  [ -f "$CKPT" ] || { echo "FATAL: TR1 snapshot never appeared"; exit 1; }
  for split in bprna_ts0 bprna_new; do
      wait_quiet
      echo "=== TR1 ff step=20000 w=-1 $split ==="
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
          "$PY" "$REPO/eval/ss/evaluate_decision.py" \
          --checkpoint "$CKPT" \
          --data "$ART/ss_data/jsonl/${split}.jsonl" \
          --embedding-split "$split" \
          --calib-data "$DEV" \
          --prior-weight -1 \
          --out "$ART/eval_decision/tr1_ff_step20000_${split}" \
          --encoder-size 35M --device cuda --head-chunk 8 \
          --tag "tr1_ff_step20000_${split}"
      echo "=== TR1 $split done rc=$? ==="
  done
  echo "########## TR1 final eval done ##########"
} >> "$LOG" 2>&1
echo detached
