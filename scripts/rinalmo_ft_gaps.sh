#!/bin/bash
# T-A46b: RiNALMo-ft (Zenodo ckpt) on the six missing board splits.
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
E=/mnt/cunyuliu/rna-jepa
CKPT=$E/refmodels/models/rinalmo_giga_ss_bprna_ft.pt
REPO=/home/cunyuliu/rna-jepa
LOG=$E/eval_decision/rinalmo_ft_gaps.log
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
echo "=== rinalmo-ft gaps start $(date +%F\ %T) on card $gpu (free ${bestfree}MB) ===" >> $LOG
for sp in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
  out=$E/eval_decision/rinalmo_ft_${sp}.json
  if [ -f "$out" ]; then continue; fi
  echo "[ft] $sp $(date +%T)" >> $LOG
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH nice -n 10 \
    $PY tools/eval_rinalmo_ft.py --ckpt "$CKPT" \
    --data $E/ss_data/jsonl/${sp}.jsonl \
    --out "$out" >> $LOG 2>&1
  echo "[ft] $sp rc=$? $(date +%T)" >> $LOG
done
echo "=== rinalmo-ft gaps done $(date +%F\ %T) ===" >> $LOG
