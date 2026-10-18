#!/bin/bash
# Speed v3 GPU readout — runs in the post-training window (chain after bias sweep).
# Measures fp16 forward (losslessness pre-verified on CPU: 120/120 bitwise, S15.65)
# at B=64 on the free card, plus a fp32 re-run for the same-window comparison.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
EVAL_DIR=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s1_step20000
export PYTHONPATH=$REPO:$REPO/src
export OMP_NUM_THREADS=12

while ! grep -q BIAS_SWEEP_DONE $EVAL_DIR/bias_sweep.log 2>/dev/null; do sleep 300; done
echo "[$(date '+%T')] card free; speed v3 window" >> $D/eval_decision/speed_v3.log
CK=$D/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt
for MODE in fp16 autocast_bf16; do
  $PY $REPO/tools/speed_v3_fp16.py --checkpoint $CK --batch-size 64 \
    --out $D/eval_decision/speed_v3_4090_${MODE}_b64.json --mode $MODE \
    >> $D/eval_decision/speed_v3.log 2>&1
done
$PY $REPO/tools/speed_batched_v2.py --checkpoint $CK --batch-size 64 \
  --out $D/eval_decision/speed_v3_4090_fp32_b64.json --decode c_batched \
  >> $D/eval_decision/speed_v3.log 2>&1
echo "[$(date '+%T')] SPEED_V3_DONE" >> $D/eval_decision/speed_v3.log
