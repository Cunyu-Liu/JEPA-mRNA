#!/bin/bash
# RiNALMo-ft 4090 re-measurement via the NON-flash path (model-native
# MultiHeadSelfAttention + rope). attention.py imports flash_attn at module
# top level, so the SDPA shim must be on PYTHONPATH to satisfy the import;
# the config patch then selects the non-flash classes at build time.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
REPO=/root/rna_jev
D=/root/rna_jepa_data
OUT=$D/eval_decision/rinalmoft_4090
CKPT=/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt
RUNNER=/root/rna_jev/tools/run_noflash.py
export PYTHONPATH=/root/rna_jev/tools/flash_shim:$REPO/src:$REPO/tools
mkdir -p $OUT
LOG=$OUT/run.log

echo "[$(date '+%T')] TS0 eval (noflash path)" >> $LOG
$PY $RUNNER --ckpt $CKPT \
  --data $D/ss_data/jsonl/bprna_ts0.jsonl \
  --out $OUT/ts0_noflash.json >> $LOG 2>&1

echo "[$(date '+%T')] TestSetB eval (noflash path)" >> $LOG
$PY $RUNNER --ckpt $CKPT \
  --data $D/ss_data/jsonl/testsetb.jsonl \
  --out $OUT/testsetb_noflash.json >> $LOG 2>&1

echo "[$(date '+%T')] RINALMOFT_NOFLASH_DONE" >> $LOG
