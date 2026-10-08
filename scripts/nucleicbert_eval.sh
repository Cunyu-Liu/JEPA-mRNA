#!/bin/bash
# T-A46e: wait for the nucleicbert TRAINING PROCESS to exit, then eval 8 splits.
# (Waiting on the ckpt file alone is wrong: best-val ckpt is saved after ep0
# while training continues; eval must use the FINAL best ckpt.)
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
E=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
CKPT=$E/refmodels/models/nucleicbert_ss_tr1c.pt
LOG=$E/eval_decision/nucleicbert_eval.log
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
cd $REPO || exit 1

while pgrep -f 'eval_nucleicbert.py train' > /dev/null; do
  sleep 60
done
echo "=== training process exited; eval start $(date +%F\ %T) ===" >> $LOG
if [ ! -f "$CKPT" ]; then
  echo "ERROR: $CKPT missing after training exit" >> $LOG
  exit 1
fi

gpu=""
bestfree=0
for c in 0 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  free=$((t - u))
  [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
done
echo "=== nucleicbert eval start $(date +%F\ %T) on card $gpu (free ${bestfree}MB) ===" >> $LOG

for sp in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
  out=$E/eval_decision/baselines_nucleicbert_${sp}.json
  if [ -f "$out" ]; then continue; fi
  echo "[nb] $sp $(date +%T)" >> $LOG
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH nice -n 10 \
    $PY tools/eval_nucleicbert.py eval --ckpt "$CKPT" --split "$sp" --out "$out" >> $LOG 2>&1
  echo "[nb] $sp rc=$? $(date +%T)" >> $LOG
done
echo "=== nucleicbert eval done $(date +%F\ %T) ===" >> $LOG
