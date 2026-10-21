#!/usr/bin/env bash
# T-A99 launcher v2: Jev (r2d) head on frozen NucleicBERT embeddings.
# Stage 1: extract NB embeddings (bprna_tr1c + board splits), waits GPU>=8GB.
# Stage 2: train r2d-family head via train_decision on NB embedding dir
#          (single variable vs r2dtr1c_b4_s0 = the embedding source).
# Stage 3: 8-split eval (exact evaluate_decision protocol, w=-1, VL0 Platt).
# Log: /root/nb_jev.log (detached nohup from an SSH-tolerant wrapper).
set -u
LOG=/root/nb_jev.log
PY=/root/miniconda3/envs/lucaone/bin/python
TOOLS=/home/rna_jev/tools
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools
E=/root/rna_jepa_data/embeddings/nucleicbert
OUT=/root/rna_jepa_data/runs/nb_jev_r2dtr1c_s0

wait_gpu () {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$FREE" -ge "$1" ] && break
    sleep 120
  done
}

echo "[ta99] $(date) stage-1 wait (8GB)" >> $LOG
wait_gpu 8000
echo "[ta99] $(date) stage-1 embeddings start" >> $LOG
cd $TOOLS
for SPLIT in bprna_tr1c bprna_vl0 bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
  $PY extract_nb_embeddings.py --split $SPLIT --batch-size 8 >> $LOG 2>&1
  echo "[ta99] $(date) emb $SPLIT rc=$?" >> $LOG
done

echo "[ta99] $(date) stage-2 wait (14GB)" >> $LOG
wait_gpu 14000
echo "[ta99] $(date) stage-2 train start" >> $LOG
cd /home/rna_jev
$PY -m rnajepa.train_decision \
  --arm decision \
  --out $OUT \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /teams/HoulabData_1788502323/liucunyu/rna_jev_data/ss_data/teacher/bprna_tr1c \
  --steps 20000 --batch-size 2 --lr 1e-4 --seed 0 \
  --head flat --scorer resnet2d \
  --embedding-dir $E --embedding-d-model 1024 \
  --device cuda >> $LOG 2>&1
echo "[ta99] $(date) train rc=$?" >> $LOG

echo "[ta99] $(date) stage-3 eval wait (8GB)" >> $LOG
wait_gpu 8000
for SPLIT in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
  $PY eval/ss/evaluate_decision.py \
    --checkpoint $OUT/resume.pt \
    --data /root/rna_jepa_data/ss_data/jsonl/$SPLIT.jsonl \
    --embedding-dir $E --embedding-d-model 1024 \
    --out /root/rna_jepa_data/eval_decision/nb_jev_$SPLIT/result.json \
    --device cuda >> $LOG 2>&1
  echo "[ta99] $(date) eval $SPLIT rc=$?" >> $LOG
done
echo "[ta99] $(date) ALL DONE" >> $LOG
