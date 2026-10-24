#!/bin/bash
# Round-28 mix arm, watchdog version: auto-restarts training from resume.pt
# whenever the process is killed (cgroup reaps sessions periodically on this
# host). Loops until step 20000 completes, then runs the 4-split eval chain.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
LOG=/root/chanrg_arm_gt.log
J=/root/rna_jepa_data/ss_data/jsonl
E=/root/rna_jepa_data/embeddings/rinalmo-giga
D=/root/rna_jepa_data
OUT=$D/runs/chanrg_r2d_mix_s0
TEACHER=$D/ss_data/teacher_mix_slim
CKPT=$D/ckpts_1014/chanrg_r2d_mix_s0_step20000.pt

echo "[mix-wd] $(date) watchdog start" >> $LOG
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools

MAX_RESTARTS=40
r=0
while [ $r -lt $MAX_RESTARTS ]; do
  if [ -f $CKPT ]; then
    echo "[mix-wd] $(date) final ckpt exists, break" >> $LOG
    break
  fi
  RESUME_ARG=""
  if [ -f $OUT/resume.pt ]; then
    RESUME_ARG="--resume $OUT/resume.pt"
  fi
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -m rnajepa.train_decision \
    --arm chanrg_r2d_mix_s0 \
    --out $OUT \
    --data $J/mixed_train.jsonl \
    --teacher-dir $TEACHER \
    --steps 20000 --batch-size 2 --lr 1e-4 --seed 0 \
    --log-every 25 --save-every 500 \
    --head flat --scorer resnet2d --gpu-reserve-gb 18 \
    --embedding-dir $E --embedding-d-model 1280 \
    --device cuda $RESUME_ARG >> $LOG 2>&1
  RC=$?
  echo "[mix-wd] $(date) train exited rc=$RC (restart #$r)" >> $LOG
  if [ -f $CKPT ]; then break; fi
  r=$((r+1))
  sleep 20
done

if [ ! -f $CKPT ]; then
  cp $OUT/resume.pt $CKPT 2>/dev/null || { echo "[mix-wd] FATAL no ckpt" >> $LOG; exit 1; }
fi
echo "[mix-wd] $(date) snapshot frozen" >> $LOG

for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  EVAL_OUT=$D/eval_decision/chanrg_mix_s0/$SPLIT
  mkdir -p $EVAL_OUT
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 $PY /home/rna_jev/eval/ss/evaluate_decision.py \
    --checkpoint $CKPT \
    --data $J/$SPLIT.jsonl \
    --out $EVAL_OUT \
    --tag chanrg_mix_s0_$SPLIT \
    --prior-weight -1 \
    --calib-data $J/chanrg_train_validation.jsonl \
    --embedding-split $SPLIT \
    --head-chunk 8 \
    --device cuda >> $LOG 2>&1
  echo "[mix-wd] $(date) eval $SPLIT rc=$?" >> $LOG
done
echo "[mix-wd] $(date) ALL DONE" >> $LOG
