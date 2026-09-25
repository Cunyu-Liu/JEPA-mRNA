#!/bin/bash
# Evaluate our checkpoints on the official full TS0 (1305 = RiNALMo/RNAformer split).
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/ts0_1305.log
EVAL_GPU="MIG-69a32e2d-cdb9-54e2-bdde-3228e0195de7"

eval_one() {
  local arm="$1" step="$2"
  local ckpt="$ART/ckpts/${arm}_step${step}.pt"
  local out="$ART/eval_decision/official1305_${arm}_step${step}"
  [ -f "$out/result.json" ] && return 0
  [ -f "$ckpt" ] || { echo "MISSING $ckpt"; return 1; }
  echo "=== OFFICIAL1305 $arm $(date +%T) ==="
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
      PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
      "$PY" "$REPO/eval/ss/evaluate_decision.py" \
      --checkpoint "$ckpt" \
      --data "$ART/ss_data/jsonl/bprna_ts0_1305.jsonl" \
      --embedding-split bprna_ts0_1305 \
      --calib-data "$DEV" \
      --prior-weight -1 \
      --out "$out" \
      --encoder-size 35M --device cuda --head-chunk 8 \
      >> "$LOG" 2>&1
  echo "=== OFFICIAL1305 $arm done rc=$? ==="
}

echo "start $(date +%T)" >> "$LOG"
eval_one rinalmo_ff_b4_s0 20000
eval_one rinalmo_big_b4_s0 20000
eval_one rinalmo_bigtr1_b4_s0 20000
echo "all done $(date +%T)" >> "$LOG"
