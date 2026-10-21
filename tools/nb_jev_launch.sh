#!/usr/bin/env bash
# T-A99 launcher v3 (fixed eval protocol after dry-run audit per §15.48 rule):
# - train_decision saves config.embedding_dir in the checkpoint, so eval must
#   NOT pass --embedding-dir (evaluate_decision reads it from the ckpt config);
#   passing it is an unknown flag -> argparse error. Removed.
# - eval aligned to watch7 protocol: --prior-weight -1 (model's own weight),
#   --calib-data VL0 (Platt, held-out; VL0 NB embeddings extracted in stage-1),
#   --head-chunk 8, --tag, archiveii uses --embedding-split archiveii
#   (full-shard name; embok_clean is a subset of it).
set -u
LOG=/root/nb_jev.log
PY=/root/miniconda3/envs/lucaone/bin/python
TOOLS=/home/rna_jev/tools
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools
E=/root/rna_jepa_data/embeddings/nucleicbert
D=/root/rna_jepa_data
OUT=$D/runs/nb_jev_r2dtr1c_s0

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
for SPLIT in bprna_tr1c bprna_vl0 bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii testsetb; do
  # NOTE: archiveii (full shard) covers archiveii_embok_clean (k7 lesson: the
  # clean subset needs the FULL archiveii shard name, not embok_clean).
  $PY extract_nb_embeddings.py --split $SPLIT --batch-size 8 >> $LOG 2>&1
  echo "[ta99] $(date) emb $SPLIT rc=$?" >> $LOG
done

echo "[ta99] $(date) stage-2 wait (14GB)" >> $LOG
wait_gpu 14000
echo "[ta99] $(date) stage-2 train start" >> $LOG
cd /home/rna_jev
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m rnajepa.train_decision \
  --arm nb_jev_r2dtr1c_s0 \
  --out $OUT \
  --data $D/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /teams/HoulabData_1788502323/liucunyu/rna_jev_data/ss_data/teacher/bprna_tr1c \
  --steps 20000 --batch-size 2 --lr 1e-4 --seed 0 \
  --log-every 25 --save-every 500 \
  --head flat --scorer resnet2d --gpu-reserve-gb 18 \
  --embedding-dir $E --embedding-d-model 1024 \
  --device cuda >> $LOG 2>&1
echo "[ta99] $(date) train rc=$?" >> $LOG

# 2b. wait until step 20000 in train_log (converge protocol = same fixed 20k
# as the r2d family arms; the family's convergence evidence lives in ledger §14.56)
while true; do
  S=$(tail -1 $OUT/train_log.jsonl 2>/dev/null | $PY -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0)
  [ "$S" -ge 20000 ] && break
  sleep 300
done
cp $OUT/resume.pt $D/ckpts_1014/nb_jev_r2dtr1c_s0_step20000.pt 2>/dev/null || true
echo "[ta99] $(date) snapshot frozen" >> $LOG

echo "[ta99] $(date) stage-3 eval wait (8GB)" >> $LOG
wait_gpu 8000
EVAL_OUT=$D/eval_decision/nb_jev_r2dtr1c_s0_step20000
mkdir -p $EVAL_OUT
for SPLIT in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
  [ -f $EVAL_OUT/$SPLIT/result.json ] && continue
  ARCH_EMB=""
  if [ "$SPLIT" = "archiveii_embok_clean" ]; then ARCH_EMB="--embedding-split archiveii"; fi
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 $PY /home/rna_jev/eval/ss/evaluate_decision.py \
    --checkpoint $OUT/resume.pt \
    --data $D/ss_data/jsonl/$SPLIT.jsonl \
    --out $EVAL_OUT/$SPLIT \
    --tag nb_jev_${SPLIT} \
    --prior-weight -1 \
    --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
    --head-chunk 8 $ARCH_EMB \
    --device cuda >> $LOG 2>&1
  echo "[ta99] $(date) eval $SPLIT rc=$?" >> $LOG
done
echo "[ta99] $(date) ALL DONE" >> $LOG
