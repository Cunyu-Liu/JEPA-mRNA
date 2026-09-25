#!/bin/bash
# TestSetB (Rivas 430) zero-shot evaluation: our 3 arms + Vienna baselines.
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/testsetb.log
EVAL_GPU="MIG-69a32e2d-cdb9-54e2-bdde-3228e0195de7"

echo "start $(date +%T)" >> "$LOG"
for arm in rinalmo_ff_b4_s0 rinalmo_big_b4_s0 rinalmo_bigtr1_b4_s0; do
  out="$ART/eval_decision/testsetb_${arm}_step20000"
  [ -f "$out/result.json" ] && continue
  ckpt="$ART/ckpts/${arm}_step20000.pt"
  [ -f "$ckpt" ] || { echo "MISSING $ckpt" >> "$LOG"; continue; }
  echo "=== TESTSETB $arm $(date +%T) ===" >> "$LOG"
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
      PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" "$REPO/eval/ss/evaluate_decision.py" \
      --checkpoint "$ckpt" \
      --data "$ART/ss_data/jsonl/testsetb.jsonl" \
      --embedding-split testsetb \
      --calib-data "$DEV" \
      --prior-weight -1 \
      --out "$out" \
      --encoder-size 35M --device cuda --head-chunk 8 \
      >> "$LOG" 2>&1
  echo "=== TESTSETB $arm done rc=$? ===" >> "$LOG"
done

# Vienna + nussinov baselines on the same file, same metric code
env PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 \
    "$PY" "$REPO/eval/ss/run_baselines.py" \
    --split testsetb \
    --baselines vienna_mfe,vienna_centroid,vienna_mea,nussinov_turner \
    --out "$ART/eval_decision/baselines_testsetb.json" >> "$LOG" 2>&1
echo "all done $(date +%T)" >> "$LOG"
