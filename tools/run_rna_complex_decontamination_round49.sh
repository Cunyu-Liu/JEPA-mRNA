#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
DATA="$ROOT/data/rna_complex_round49"
cd /home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
while ! $PY tools/audit_rna_complex_round49.py \
  --entries-root "$DATA/entries" --expected-ids-json "$DATA/rcsb_rna_protein_all.json" \
  --pretrain-fasta "$ROOT/source_corpora/structrfm_round48/pretraining.fasta" \
  --clean-splits "$ROOT/data/structrfm_pretrain_clean" \
  --train-jsonl /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --vl0-jsonl "$ROOT/data/validation/vl0_196.jsonl" \
  --out-dir "$DATA/decontamination" --validate-collection-only \
  > "$ROOT/logs/round48/rna_complex_collection_gate.log" 2>&1; do sleep 60; done
$PY tools/audit_rna_complex_round49.py \
  --entries-root "$DATA/entries" --expected-ids-json "$DATA/rcsb_rna_protein_all.json" \
  --pretrain-fasta "$ROOT/source_corpora/structrfm_round48/pretraining.fasta" \
  --clean-splits "$ROOT/data/structrfm_pretrain_clean" \
  --train-jsonl /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --vl0-jsonl "$ROOT/data/validation/vl0_196.jsonl" \
  --out-dir "$DATA/decontamination" \
  --threads 8 > "$ROOT/logs/round48/rna_complex_decontamination.log" 2>&1
/root/miniconda3/envs/lucaone/bin/python tools/verify_rna_complex_decontamination_round49.py \
  --out-dir "$DATA/decontamination" > "$ROOT/logs/round48/rna_complex_verification.log" 2>&1
