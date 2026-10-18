#!/bin/bash
# After r2dtr1c_b4_s1 finishes + its auto-eval completes, run:
#   1. xens2 2-seed board protocol reproduction (plana s0+s1 bucket x NEW r2d s0
#      ... note: board xens2 uses r2d s0 as the cross-family member; the NEW s1
#      arm enables (a) r2d-family 2-seed robustness row, (b) TS2-focused
#      ensemble rescans with alternative members).
#   2. TS2 rescans: w sweep {0.5,0.6,0.7,0.8} x r2d {s0, s1, 0.5*s0+0.5*s1}.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
CK=$D/ckpts_1014
TAG=rinalmo_r2dtr1c_b4_s1
EVAL_DIR=$D/eval_decision/ow_${TAG}_step20000
export PYTHONPATH=$REPO/src:$REPO/tools

# wait for auto-eval to finish all 8 splits
while true; do
  DONE=$(grep -c ALL_EVALS_DONE $EVAL_DIR/auto_eval.log 2>/dev/null || echo 0)
  [ "$DONE" -ge 1 ] && break
  sleep 600
done
echo "[$(date '+%T')] auto-eval done, starting xens2 resweep" >> $EVAL_DIR/resweep.log

# 1. board-protocol xens2 reproduction on TS0 (2-seed plana bucket x r2d s0)
$PY $REPO/tools/xens2_eval.py \
  --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt \
  --plana-ckpt2 $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
  --r2d-ckpt $CK/rinalmo_r2dtr1c_b4_s0_step20000.pt \
  --data $D/ss_data/jsonl/bprna_ts0.jsonl \
  --embedding-split bprna_ts0 \
  --out $EVAL_DIR/xens2_board_ts0_repro.json \
  --device cuda >> $EVAL_DIR/resweep.log 2>&1

# 2. TS2 rescans: r2d member = new s1 (single) at several w
for W in 0.5 0.6 0.7 0.8; do
  $PY $REPO/tools/xens2_eval.py \
    --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt \
    --plana-ckpt2 $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
    --r2d-ckpt $CK/${TAG}_step20000.pt \
    --data $D/ss_data/jsonl/ref_pdb_ts2.jsonl \
    --embedding-split ref_pdb_ts2 \
    --w-plana $W \
    --out $EVAL_DIR/xens2_ts2_w${W}_r2ds1.json \
    --device cuda >> $EVAL_DIR/resweep.log 2>&1
done
echo "[$(date '+%T')] RESWEEP_DONE" >> $EVAL_DIR/resweep.log
