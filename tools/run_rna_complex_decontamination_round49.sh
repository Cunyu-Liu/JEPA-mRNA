#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
DATA="$ROOT/data/rna_complex_round49"
while [[ ! -f "$DATA/entries/collection.json" ]]; do sleep 60; done
cd /home/rna_jev
/root/miniconda3/envs/lucaone/bin/python tools/audit_rna_complex_round49.py \
  --entries-root "$DATA/entries" \
  --pretrain-fasta "$ROOT/source_corpora/structrfm_round48/pretraining.fasta" \
  --clean-splits "$ROOT/data/structrfm_pretrain_clean" \
  --train-jsonl /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --vl0-jsonl "$ROOT/data/validation/vl0_196.jsonl" \
  --out-dir "$DATA/decontamination" \
  --threads 8 > "$ROOT/logs/round48/rna_complex_decontamination.log" 2>&1
