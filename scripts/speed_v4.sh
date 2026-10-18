#!/bin/bash
# Speed v4: push batch dimension further with fp16 weights.
# v3 ladder: B=64 fp16 = 119.55 seq/s. Test B=128, B=192, B=256 fp16.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
OUT=$D/eval_decision
export PYTHONPATH=$REPO/src
LOG=$OUT/speed_v4.log
for B in 128 192 256; do
  [ -f $OUT/speed_v4_4090_fp16_b${B}.json ] && continue
  echo "[$(date '+%T')] v4 fp16 B=$B" >> $LOG
  CUDA_VISIBLE_DEVICES=0 $PY $REPO/tools/speed_v3_fp16.py \
    --checkpoint $D/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt \
    --batch-size $B \
    --out $OUT/speed_v4_4090_fp16_b${B}.json \
    >> $LOG 2>&1 || echo "[$(date '+%T')] B=$B FAILED (likely OOM)" >> $LOG
done
echo "[$(date '+%T')] SPEED_V4_DONE" >> $LOG
