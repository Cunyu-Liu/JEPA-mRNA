#!/bin/bash
# MRL dual-head watcher: run when GPU has >=10GB free (embeddings+heads).
set -u
LOG=/root/mrl_dual.log
PY=/root/miniconda3/envs/lucaone/bin/python

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  [ "$FREE" -ge 10000 ] && break
  sleep 120
done
echo "[mrl] $(date) GPU free ${FREE}MiB -> running" >> $LOG
cd /home/rna_jev/tools
PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools \
  $PY eval_mrl_dual.py --device cuda >> $LOG 2>&1
echo "[mrl] $(date) DONE rc=$?" >> $LOG
