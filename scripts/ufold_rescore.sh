#!/bin/bash
# T-A46d: score the 5 new UFold gap dbn files through the unified project
# scorer (run_baselines.py --external-dbn), exactly like the 3 existing
# baselines_ufold_*.json rows; plus RiNALMo-ft on board TS0 (bprna_ts0).
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
E=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
LOG=$E/eval_decision/ufold_score.log
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
cd $REPO || exit 1

echo "=== ufold rescore start $(date +%F\ %T) ===" >> $LOG
for sp in testsetb ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean; do
  out=$E/eval_decision/baselines_ufold_${sp}.json
  if [ -f "$out" ]; then continue; fi
  dbn=$E/eval_decision/ufold_${sp}.dbn
  echo "[rescore] $sp $(date +%T)" >> $LOG
  $PY eval/ss/run_baselines.py --split "$sp" \
    --baselines "" \
    --external-dbn "$dbn" --external-name ufold \
    --out "$out" >> $LOG 2>&1
  echo "[rescore] $sp rc=$? $(date +%T)" >> $LOG
done

# RiNALMo-ft on board TS0 split (bprna_ts0, n=1288; existing runs used the
# official 1305 file, this is the board's own split file)
out=$E/eval_decision/rinalmo_ft_bprna_ts0.json
if [ ! -f "$out" ]; then
  gpu=""
  bestfree=0
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$((t - u))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  echo "[ft-ts0] card $gpu free ${bestfree}MB $(date +%T)" >> $LOG
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH nice -n 10 \
    $PY tools/eval_rinalmo_ft.py --ckpt $E/refmodels/models/rinalmo_giga_ss_bprna_ft.pt \
    --data $E/ss_data/jsonl/bprna_ts0.jsonl \
    --out "$out" >> $LOG 2>&1
  echo "[ft-ts0] rc=$? $(date +%T)" >> $LOG
fi
echo "=== ufold rescore done $(date +%F\ %T) ===" >> $LOG
