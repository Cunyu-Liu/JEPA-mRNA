#!/bin/bash
# Wait for ff s2 snapshot at step 20000, then eval (serial protocol).
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
TEST=$ART/ss_data/jsonl/bprna_ts0.jsonl
LOG=$ART/eval_decision/s2_eval.log
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
CKPT="$ART/ckpts/rinalmo_ff_b4_s2_step20000.pt"

{
  for _ in $(seq 1 2400); do
      [ -f "$CKPT" ] && break
      sleep 60
  done
  [ -f "$CKPT" ] || { echo "FATAL: snapshot never appeared"; exit 1; }
  for _ in $(seq 1 1440); do
      pgrep -f "[e]valuate_decision.py" >/dev/null && { sleep 30; continue; }
      pgrep -f "[r]un_s1_casc_eval.sh" >/dev/null && { sleep 30; continue; }
      break
  done
  echo "=== s2 step=20000 w=-1 ts0 ==="
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
      PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" "$REPO/eval/ss/evaluate_decision.py" \
      --checkpoint "$CKPT" \
      --data "$TEST" \
      --calib-data "$DEV" \
      --prior-weight -1 \
      --out "$ART/eval_decision/arms_rinalmo_ff_b4_s2_step20000_ts0" \
      --encoder-size 35M --device cuda --head-chunk 8 \
      --tag "arms_rinalmo_ff_b4_s2_step20000_ts0"
  echo "=== s2 done rc=$? ==="
} >> "$LOG" 2>&1
echo detached
