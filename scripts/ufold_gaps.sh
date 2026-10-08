#!/bin/bash
# T-A46c: UFold gap fills on the five missing board splits.
# Prior runs used the lucaone env (UFold deps live there).
set -u
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
E=/mnt/cunyuliu/rna-jepa
LOG=$E/eval_decision/ufold_gaps.log

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
cd $REPO || exit 1

gpu=""
bestfree=0
for c in 0 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  free=$((t - u))
  [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
done
echo "=== ufold gaps start $(date +%F\ %T) on card $gpu (free ${bestfree}MB) ===" >> $LOG

for sp in testsetb ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean; do
  out=$E/eval_decision/baselines_ufold_${sp}.json
  if [ -f "$out" ]; then continue; fi
  echo "[ufold] $sp $(date +%T)" >> $LOG
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH nice -n 10 \
    $PY eval/ss/run_ufold.py --split "$sp" --eval-mode >> $LOG 2>&1
  echo "[ufold] $sp rc=$? $(date +%T)" >> $LOG
done
echo "=== ufold gaps done $(date +%F\ %T) ===" >> $LOG
