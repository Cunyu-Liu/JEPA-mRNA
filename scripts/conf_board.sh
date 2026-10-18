#!/bin/bash
# Confidence 8-split expansion: run predict_with_confidence on every board
# split (CPU is fine, head-only 9.3M), then compute per-split ECE_struct
# (raw + VL0-recalibrated) for the preprint's confidence claim.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
OUT=$D/eval_decision/confidence_board
CK=$D/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt
export PYTHONPATH=$REPO/src:$REPO/tools
mkdir -p $OUT
SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb"
for sp in $SPLITS; do
  [ -f $OUT/$sp.json ] && continue
  echo "[$(date '+%T')] conf $sp" >> $OUT/board_conf.log
  $PY $REPO/tools/predict_with_confidence.py \
    --checkpoint $CK \
    --data $D/ss_data/jsonl/${sp}.jsonl \
    --out $OUT/$sp.json \
    --device cuda --gt-f1 >> $OUT/board_conf.log 2>&1
done
echo "[$(date '+%T')] CONF_BOARD_DONE" >> $OUT/board_conf.log
