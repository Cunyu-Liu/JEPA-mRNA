#!/bin/bash
# RiNALMo-ft 4090 same-node re-measurement (T-A71):
#   1. TS0 clean protocol (accuracy row)
#   2. Speed reading on TS0 (same-node reference for the throughput table)
#   3. TestSetB (clean-corpus comparison row)
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
OUT=$D/eval_decision/rinalmoft_4090
CKPT=/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt
export PYTHONPATH=/root/rna_jev/tools/flash_shim:$REPO/src:$REPO/tools
mkdir -p $OUT
LOG=$OUT/run.log

echo "[$(date '+%T')] TS0 eval" >> $LOG
$PY $REPO/tools/eval_rinalmo_ft.py --ckpt $CKPT \
  --data $D/ss_data/jsonl/bprna_ts0.jsonl \
  --out $OUT/ts0.json >> $LOG 2>&1

echo "[$(date '+%T')] TestSetB eval" >> $LOG
$PY $REPO/tools/eval_rinalmo_ft.py --ckpt $CKPT \
  --data $D/ss_data/jsonl/testsetb.jsonl \
  --out $OUT/testsetb.json >> $LOG 2>&1

echo "[$(date '+%T')] RINALMOFT_4090_DONE" >> $LOG
