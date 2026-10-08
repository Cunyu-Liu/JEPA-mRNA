#!/bin/bash
# T-A46f: EternaFold gap fills, per-sequence contrafold predict.
# contrafold predict rejects multi-seq FASTA with unequal lengths
# ("ERROR: Not all sequences have the same length"), so we split to
# per-sequence files and run 32 parallel contrafold processes (96 cores).
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
E=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
LOG=$E/eval_decision/eterna_gaps2.log
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
cd $REPO || exit 1

mkdir -p $E/eval_decision/fasta /tmp/eterna_jobs
SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean"

echo "=== eterna per-seq start $(date +%F\ %T) ===" >> $LOG
for sp in $SPLITS; do
  out=$E/eval_decision/baselines_eternafold_${sp}.json
  if [ -f "$out" ]; then continue; fi
  fa=$E/eval_decision/fasta/eterna_${sp}.fa
  dbn=$E/eval_decision/fasta/eterna_${sp}.dbn
  echo "[eterna] $sp: prepare $(date +%T)" >> $LOG
  $PY scripts/eterna_prep.py "$sp" "$fa" "/tmp/eterna_jobs/${sp}" >> $LOG 2>&1
  echo "[eterna] $sp: per-seq predict start $(date +%T)" >> $LOG
  find /tmp/eterna_jobs/${sp} -name '*.fa' -print0 |
    xargs -0 -P 32 -n 1 bash /home/cunyuliu/rna-jepa/scripts/eterna_one.sh >> $LOG 2>&1
  $PY scripts/eterna_cat.py "/tmp/eterna_jobs/${sp}" "$dbn" >> $LOG 2>&1
  echo "[eterna] $sp: scoring $(date +%T)" >> $LOG
  $PY eval/ss/run_baselines.py --split "$sp" \
    --baselines "" \
    --external-dbn "$dbn" --external-name eternafold \
    --out "$out" >> $LOG 2>&1
  echo "[eterna] $sp rc=$? $(date +%T)" >> $LOG
  rm -rf /tmp/eterna_jobs/${sp}
done
echo "=== eterna per-seq done $(date +%F\ %T) ===" >> $LOG
