#!/bin/bash
# Extended TS2 attack after the base resweep: decode-bias sweep (S15.19 lever:
# recall-type bias on the ensemble score) with the NEW r2d s1 member.
# Rationale: TS2 gap decomposition showed P 0.935 / R 0.794 — the deficit is
# recall on rare short stems; a small +decode bias trades P for R.
# Protocol: VL0-gated selection (bias chosen on VL0, not on TS2 itself).
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
CK=$D/ckpts_1014
EVAL_DIR=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s1_step20000
export PYTHONPATH=$REPO/src:$REPO/tools

# wait for base resweep to complete
while ! grep -q RESWEEP_DONE $EVAL_DIR/resweep.log 2>/dev/null; do sleep 300; done
echo "[$(date '+%T')] base resweep done; starting VL0 bias-gate + TS2 bias sweep" >> $EVAL_DIR/bias_sweep.log

# 1. VL0 gate: pick the bias on VL0 (never on TS2)
$PY $REPO/tools/xens_bias_sweep.py \
  --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
  --r2d-ckpt $CK/rinalmo_r2dtr1c_b4_s1_step20000.pt \
  --data $D/ss_data/jsonl/bprna_vl0.jsonl \
  --embedding-split bprna_vl0 \
  --biases "0,0.5,1,1.5,2" \
  --out $EVAL_DIR/bias_gate_vl0.json \
  --device cuda >> $EVAL_DIR/bias_sweep.log 2>&1

# 2. TS2 with the VL0-chosen bias range (report the full curve, quote only
#    the VL0-selected point in any headline)
$PY $REPO/tools/xens_bias_sweep.py \
  --plana-ckpt $D/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt \
  --r2d-ckpt $CK/rinalmo_r2dtr1c_b4_s1_step20000.pt \
  --data $D/ss_data/jsonl/ref_pdb_ts2.jsonl \
  --embedding-split ref_pdb_ts2 \
  --biases "0,0.5,1,1.5,2" \
  --out $EVAL_DIR/bias_sweep_ts2.json \
  --device cuda >> $EVAL_DIR/bias_sweep.log 2>&1
echo "[$(date '+%T')] BIAS_SWEEP_DONE" >> $EVAL_DIR/bias_sweep.log
