#!/bin/bash
# xens2 board protocol (w=0.7, plana s0+s1 bucket, r2d = new s1) full 8-split
# reproduction on the 4090 node. Extends the v3.20 reproduction gate from
# TS0-only to the whole board. Writes one JSON per split.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
CK=$D/ckpts_1014
OUT=$D/eval_decision/xens2_board_4090
export PYTHONPATH=$REPO/src:$REPO/tools
mkdir -p $OUT

SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb"
for sp in $SPLITS; do
  [ -f $OUT/$sp.json ] && continue
  EMB="--embedding-split $sp"
  if [ "$sp" = "archiveii_embok_clean" ]; then EMB="--embedding-split archiveii"; fi
  echo "[$(date '+%T')] board repro $sp" >> $OUT/board_repro.log
  $PY $REPO/tools/xens2_eval.py \
    --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt \
    --plana-ckpt2 $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
    --r2d-ckpt $CK/rinalmo_r2dtr1c_b4_s0_step20000.pt \
    --data $D/ss_data/jsonl/${sp}.jsonl \
    $EMB \
    --w-plana 0.7 \
    --out $OUT/$sp.json \
    --device cuda >> $OUT/board_repro.log 2>&1
done
echo "[$(date '+%T')] BOARD_8SPLIT_REPRO_DONE" >> $OUT/board_repro.log
