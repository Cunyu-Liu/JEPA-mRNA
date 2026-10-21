#!/usr/bin/env bash
# Watcher for the ncrna dual-head eval: waits for GPU0 free >= 6GB
# (embedding extraction batch 16 + head training), runs it detached.
set -u
LOG=/root/ncrna_dual.log
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  if [ "$FREE" -ge 6000 ]; then
    break
  fi
  sleep 120
done
echo "[watch] $(date) free=${FREE}MiB -> launching ncrna dual-head eval" >> $LOG

cd /home/rna_jev/tools
$PY eval_ncrna_dual.py --device cuda --out /root/rna_jepa_data/eval_decision/ncrna_dual_result.json >> $LOG 2>&1
echo "[watch] $(date) DONE rc=$?" >> $LOG
