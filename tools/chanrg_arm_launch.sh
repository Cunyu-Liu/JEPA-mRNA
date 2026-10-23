#!/usr/bin/env bash
# Round-27 GenF SOTA arm: Jev r2d head trained on CHANRG Train (50-200nt).
# Single variable vs rinalmo_r2dtr1c_b4_s0 = corpus (bpRNA tr1c -> chanrg_train).
# Everything else identical: frozen rinalmo-giga embeddings, flat head + resnet2d
# scorer, 20k steps, B=2, lr 1e-4, seed 0, same eval protocol.
# Stages: 1) embeddings (train+val) 2) teacher labels 3) train 4) 4-split eval.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
LOG=/root/chanrg_arm.log
J=/root/rna_jepa_data/ss_data/jsonl
E=/root/rna_jepa_data/embeddings/rinalmo-giga
D=/root/rna_jepa_data
OUT=$D/runs/chanrg_r2d_s0
TEACHER=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/ss_data/teacher/chanrg_train
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools

echo "[arm] $(date) stage-1 embeddings" >> $LOG
cd /home/rna_jev/tools
for SPLIT in chanrg_train_train chanrg_train_validation; do
  if [ ! -f $E/$SPLIT.shard0of1.npz ]; then
    $PY extract_rinalmo_embeddings.py --split $SPLIT --batch-size 8 >> $LOG 2>&1
    echo "[arm] $(date) emb $SPLIT rc=$?" >> $LOG
  fi
done

echo "[arm] $(date) stage-2 teacher labels (95k seqs, 12 proc)" >> $LOG
if [ ! -d $TEACHER ] || [ -z "$(ls -A $TEACHER 2>/dev/null)" ]; then
  cd /home/rna_jev
  $PY data/ss/generate_teacher_labels.py \
    --data $J/chanrg_train_train.jsonl \
    --out $TEACHER >> $LOG 2>&1
  echo "[arm] $(date) teacher rc=$?" >> $LOG
fi

echo "[arm] $(date) stage-3 train (20k steps)" >> $LOG
cd /home/rna_jev
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m rnajepa.train_decision \
  --arm chanrg_r2d_s0 \
  --out $OUT \
  --data $J/chanrg_train_train.jsonl \
  --teacher-dir $TEACHER \
  --steps 20000 --batch-size 2 --lr 1e-4 --seed 0 \
  --log-every 25 --save-every 500 \
  --head flat --scorer resnet2d --gpu-reserve-gb 18 \
  --embedding-dir $E --embedding-d-model 1280 \
  --device cuda >> $LOG 2>&1
RC=$?
echo "[arm] $(date) train rc=$RC" >> $LOG
[ $RC -ne 0 ] && { echo "[arm] FATAL train failed" >> $LOG; exit 1; }

cp $OUT/resume.pt $D/ckpts_1014/chanrg_r2d_s0_step20000.pt 2>/dev/null || true
echo "[arm] $(date) snapshot frozen" >> $LOG

echo "[arm] $(date) stage-4 4-split eval" >> $LOG
for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  EVAL_OUT=$D/eval_decision/chanrg_arm/$SPLIT
  mkdir -p $EVAL_OUT
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 $PY /home/rna_jev/eval/ss/evaluate_decision.py \
    --checkpoint $OUT/resume.pt \
    --data $J/$SPLIT.jsonl \
    --out $EVAL_OUT \
    --tag chanrg_arm_$SPLIT \
    --prior-weight -1 \
    --calib-data $J/chanrg_train_validation.jsonl \
    --embedding-split $SPLIT \
    --head-chunk 8 \
    --device cuda >> $LOG 2>&1
  echo "[arm] $(date) eval $SPLIT rc=$?" >> $LOG
done
echo "[arm] $(date) ALL DONE" >> $LOG
