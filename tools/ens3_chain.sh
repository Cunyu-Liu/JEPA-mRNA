#!/bin/bash
# Round-28: three-seed ensemble eval chain (s0+s1+s2 GT arms, ALL DONE at step 20000).
# Uses eval_multi_head_ensemble.py (N-head average) in place of a single model.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
LOG=/root/chanrg_arm_gt.log
J=/root/rna_jepa_data/ss_data/jsonl
D=/root/rna_jepa_data
C0=$D/ckpts_1014/chanrg_r2d_s0_gt_step20000.pt
C1=$D/ckpts_1014/chanrg_r2d_s1_gt_step20000.pt
C2=$D/ckpts_1014/chanrg_r2d_s2_gt_step20000.pt
for C in $C0 $C1 $C2; do
  if [ ! -f $C ]; then
    R=${C/ckpts_1014\//runs/}; R=${R/_step20000.pt/\/resume.pt}
    if [ -f $R ]; then C_OVERRIDE="$C_OVERRIDE $R"; else echo "[ens3] MISSING $C" >> $LOG; exit 1; fi
  fi
done

echo "[ens3] $(date) chain start" >> $LOG
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools
for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  EVAL_OUT=$D/eval_decision/chanrg_ens3_gt/$SPLIT
  mkdir -p $EVAL_OUT
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 $PY /home/rna_jev/tools/eval_multi_head_ensemble.py \
    --ckpts $C0 $C1 $C2 \
    -- \
    --data $J/$SPLIT.jsonl \
    --out $EVAL_OUT \
    --tag chanrg_ens3_gt_$SPLIT \
    --prior-weight -1 \
    --calib-data $J/chanrg_train_validation.jsonl \
    --embedding-split $SPLIT \
    --head-chunk 8 \
    --device cuda >> $LOG 2>&1
  echo "[ens3] $(date) eval $SPLIT rc=$?" >> $LOG
done
echo "[ens3] $(date) ALL DONE" >> $LOG
