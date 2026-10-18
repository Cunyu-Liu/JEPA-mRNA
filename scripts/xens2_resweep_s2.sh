#!/bin/bash
# xens2 TS2 rescan with r2d member = NEW s2 (precision-type seed),
# plus a 2-seed r2d member (0.5*s1+0.5*s2 avg via two-member option if
# supported, else single-member runs). Board protocol w=0.7 + w sweep.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
CK=$D/ckpts_1014
EVAL_DIR=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s2_step20000
export PYTHONPATH=$REPO/src:$REPO/tools

# TS2 rescan: r2d member = new s2, w sweep
for W in 0.5 0.6 0.7 0.8; do
  $PY $REPO/tools/xens2_eval.py \
    --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt \
    --plana-ckpt2 $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
    --r2d-ckpt $CK/rinalmo_r2dtr1c_b4_s2_step20000.pt \
    --data $D/ss_data/jsonl/ref_pdb_ts2.jsonl \
    --embedding-split ref_pdb_ts2 \
    --w-plana $W \
    --out $EVAL_DIR/xens2_ts2_w${W}_r2ds2.json \
    --device cuda >> $EVAL_DIR/resweep_s2.log 2>&1
done

# TS0 sanity row at board protocol w=0.7 (r2d member = new s2)
$PY $REPO/tools/xens2_eval.py \
  --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt \
  --plana-ckpt2 $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
  --r2d-ckpt $CK/rinalmo_r2dtr1c_b4_s2_step20000.pt \
  --data $D/ss_data/jsonl/bprna_ts0.jsonl \
  --embedding-split bprna_ts0 \
  --w-plana 0.7 \
  --out $EVAL_DIR/xens2_ts0_w0.7_r2ds2.json \
  --device cuda >> $EVAL_DIR/resweep_s2.log 2>&1

echo "[$(date '+%T')] S2_RESWEEP_DONE" >> $EVAL_DIR/resweep_s2.log
