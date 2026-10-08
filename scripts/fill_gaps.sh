#!/bin/bash
# T-A46: fill the baseline-board gaps (same scorer/GT/split protocol as all
# existing rows). Runs on one GPU-light card; EternaFold is CPU (contrafold).
# Splits covered per system ONLY where the board shows "—".
set -u
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/eval
E=/mnt/cunyuliu/rna-jepa
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
MXPY=/home/cunyuliu/miniconda3/envs/rna_baselines/bin/python
LOG=$E/eval_decision/fill_gaps.log
cd /home/cunyuliu/rna-jepa || exit 1
mkdir -p $E/eval_decision/fasta

ALL="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb"

echo "=== gap-fill start $(date +%F\ %T) ===" >> $LOG

# 1) Vienna family + nussinov on archiveii_embok_clean (missing cell)
if [ ! -f $E/eval_decision/baselines_archiveii_embok_clean.json ]; then
  echo "[gap] vienna+nussinov archiveii_embok_clean" >> $LOG
  $PY eval/ss/run_baselines.py --split archiveii_embok_clean \
    --out $E/eval_decision/baselines_archiveii_embok_clean.json >> $LOG 2>&1
fi

# 2) EternaFold on all splits except testsetb (already done)
EF=/mnt/cunyuliu/rna_baselines_src/EternaFold/src/contrafold
EFP=/mnt/cunyuliu/rna_baselines_src/EternaFold/parameters/EternaFoldParams.v1
for sp in $ALL; do
  testsetb_skip=0
  if [ "$sp" == "testsetb" ]; then testsetb_skip=1; fi
  if [ $testsetb_skip -eq 1 ]; then continue; fi
  out=$E/eval_decision/eternafold_${sp}.json
  if [ -f "$out" ]; then continue; fi
  fa=$E/eval_decision/fasta/eterna_${sp}.fa
  # build fasta from jsonl
  $PY - "$sp" "$fa" << 'PYEOF' >> $LOG 2>&1
import json, sys
sp, fa = sys.argv[1], sys.argv[2]
with open(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl") as fh, open(fa, "w") as out:
    for i, line in enumerate(fh):
        r = json.loads(line)
        seq = str(r["seq"]).upper().replace("T", "U")
        out.write(f">s{i}\n{seq}\n")
print(f"fasta {fa} written")
PYEOF
  dbn=$E/eval_decision/fasta/eterna_${sp}.dbn
  echo "[gap] eternafold $sp" >> $LOG
  (cd /mnt/cunyuliu/rna_baselines_src/EternaFold && \
   ./src/contrafold predict "$fa" --params parameters/EternaFoldParams.v1 > "$dbn") >> $LOG 2>&1
  $PY eval/ss/run_baselines.py --split "$sp" \
    --external-dbn "$dbn" --external-name eternafold \
    --out "$out" >> $LOG 2>&1
done

# 3) MXfold2 on missing splits (CLI in rna_baselines env)
for sp in $ALL; do
  out=$E/eval_decision/baselines_mxfold2_${sp}.json
  if [ -f "$out" ]; then continue; fi
  fa=$E/eval_decision/fasta/mxfold2_gap_${sp}.fa
  if [ ! -f "$fa" ]; then
    $PY - "$sp" "$fa" << 'PYEOF' >> $LOG 2>&1
import json, sys
sp, fa = sys.argv[1], sys.argv[2]
with open(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl") as fh, open(fa, "w") as out:
    for i, line in enumerate(fh):
        r = json.loads(line)
        seq = str(r["seq"]).upper().replace("T", "U")
        out.write(f">s{i}\n{seq}\n")
print(f"fasta {fa} written")
PYEOF
  fi
  dbn=$E/eval_decision/fasta/mxfold2_gap_${sp}.dbn
  echo "[gap] mxfold2 $sp" >> $LOG
  $MXPY -m mxfold2 predict "$fa" > "$dbn" 2>> $LOG
  $PY eval/ss/run_baselines.py --split "$sp" \
    --external-dbn "$dbn" --external-name mxfold2 \
    --out "$out" >> $LOG 2>&1
done

echo "=== gap-fill done $(date +%F\ %T) ===" >> $LOG
