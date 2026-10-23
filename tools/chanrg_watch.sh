#!/bin/bash
# CHANRG watcher: wait for GPU headroom (>=8GB) alongside nb_jev training,
# then run chanrg_board_eval.py for r2d + xens2 on 4 splits (sequential, GPU).
# Embedding extraction for CHANRG splits happens inside evaluate_decision via
# --embedding-split (falls back to in-loop forward if shard missing).
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
LOG=/root/chanrg_eval.log

wait_gpu () {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$FREE" -ge "$1" ] && break
    sleep 120
  done
}

echo "[chanrg] $(date) watcher up, waiting 8GB" >> $LOG
wait_gpu 8000
echo "[chanrg] $(date) GPU free ${FREE}MiB -> running 4-split x 2-system eval" >> $LOG
cd /home/rna_jev
PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools \
  $PY /home/rna_jev/tools/chanrg_board_eval.py --systems r2d,xens2 >> $LOG 2>&1
echo "[chanrg] $(date) DONE rc=$?" >> $LOG

# NB-ft archiveii gap (OOM earlier): rerun now if still missing
CKPT_DIR=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft
if [ ! -f $CKPT_DIR/baselines_nucleicbertft_archiveii_embok_clean.json ]; then
  echo "[chanrg] $(date) rerunning NB-ft archiveii (gap fill)" >> $LOG
  export NB_CKPT=$CKPT_DIR/pretrained.pt
  export PYTHONPATH=/root/rna_jepa_data/hf_home/nucleicbert-code:/home/rna_jev/tools
  cd /home/rna_jev/tools
  $PY eval_nucleicbert_ft.py eval --ckpt $CKPT_DIR/nb_finetuned.pt \
    --split archiveii_embok_clean \
    --out $CKPT_DIR/baselines_nucleicbertft_archiveii_embok_clean.json >> $LOG 2>&1
  echo "[chanrg] $(date) NB-ft archiveii rc=$?" >> $LOG
fi
echo "[chanrg] $(date) ALL DONE" >> $LOG
