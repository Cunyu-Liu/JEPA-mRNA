#!/bin/bash
# Physical baselines on the newly added reference splits.
#
# The benchmark was previously two splits (bprna_ts0 + bpRNA-new).  It is now
# nine, and a split with no baseline number is not evidence of anything: a model
# F1 can only be read against what a parameter-free physical model gets there.
# These are the splits added from the RNAformer reference release:
#
#   ref_pdb_ts1 / ref_pdb_ts2 / ref_pdb_ts3   experimental labels (PDB)
#   ref_pdb_ts_hard                           adversarial subset
#   ref_synthetic_test / ref_synthetic_valid   the release's own synthetic sets
#   ref_bprna_ts0                             same split as our .bpseq TS0 --
#                                             a free cross-check between two
#                                             independent readers of one dataset
#
# Runs strictly serially.  The box is shared and already carries other users'
# load; the earlier lesson was that concurrent single-core jobs only make each
# other slower.  Each split's result goes to its own file so a partial run still
# yields usable numbers.
#
# Usage: setsid nohup bash scripts/run_ref_split_baselines.sh > ... 2>&1 < /dev/null
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
OUT="${RNAJEPA_EVAL:-/mnt/cunyuliu/rna-jepa/eval_decision}"
PY="${RNAJEV_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"
export PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval
export TMPDIR=/mnt/cunyuliu/tmp
cd "$ROOT"

SPLITS="ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard ref_synthetic_test ref_synthetic_valid ref_bprna_ts0"

for split in $SPLITS; do
  jsonl="/mnt/cunyuliu/rna-jepa/ss_data/jsonl/${split}.jsonl"
  if [ ! -s "$jsonl" ]; then
    echo "### SKIP $split: no corpus at $jsonl"
    continue
  fi
  echo "### $split  ($(date +%H:%M:%S))"
  "$PY" eval/ss/run_baselines.py --split "$split" \
      --out "$OUT/baselines_${split}.json" 2>&1 | tail -14
  echo "### $split done rc=$?  ($(date +%H:%M:%S))"
done

echo "########## ref split baselines done $(date +%H:%M:%S) ##########"