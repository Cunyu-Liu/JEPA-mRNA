#!/bin/bash
# Three-seed CHANRG ensemble evaluation with artifact-level completion checks.
set -euo pipefail
PY=/root/miniconda3/envs/lucaone/bin/python
WORK=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
J=/root/rna_jepa_data/ss_data/jsonl
OUTROOT=$WORK/eval_decision
LOG=$WORK/logs/chanrg_ens3_gt.log
C0=$WORK/weights/chanrg_r2d_s0_gt_step20000.pt
C1=$WORK/weights/chanrg_r2d_s1_gt_step20000.pt
C2=$WORK/weights/chanrg_r2d_s2_gt_step20000.pt
mkdir -p "$WORK/logs" "$WORK/audits"

for C in "$C0" "$C1" "$C2"; do
  if [ ! -s "$C" ]; then
    echo "[ens3] MISSING $C" >> "$LOG"
    exit 1
  fi
done

echo "[ens3] $(date) chain start" >> "$LOG"
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools
for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  EVAL_OUT=$OUTROOT/chanrg_ens3_gt/$SPLIT
  mkdir -p "$EVAL_OUT"
  if [ -s "$EVAL_OUT/result.json" ]; then
    echo "[ens3] $(date) $SPLIT cache hit; verification follows" >> "$LOG"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 "$PY" /home/rna_jev/tools/eval_multi_head_ensemble.py \
    --ckpts "$C0" "$C1" "$C2" \
    -- \
    --data "$J/$SPLIT.jsonl" \
    --out "$EVAL_OUT" \
    --tag "chanrg_ens3_gt_$SPLIT" \
    --prior-weight -1 \
    --calib-data "$J/chanrg_train_validation.jsonl" \
    --embedding-split "$SPLIT" \
    --head-chunk 8 \
    --device cuda >> "$LOG" 2>&1
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["device"] == "cuda"; assert d["n_sequences"] == len(d["per_sequence"]); assert d["n_sequences"] > 0' \
    "$EVAL_OUT/result.json"
  echo "[ens3] $(date) eval $SPLIT artifact OK" >> "$LOG"
done

"$PY" /home/rna_jev/tools/verify_chanrg_ens3.py \
  --eval-root "$OUTROOT" \
  --require-s2 \
  --out "$WORK/audits/chanrg_ens3_verification.json" >> "$LOG" 2>&1
echo "[ens3] $(date) ALL_DONE verification PASS" >> "$LOG"
