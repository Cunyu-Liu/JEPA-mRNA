#!/bin/bash
# Build validated CHANRG train/control assets without overwriting final outputs.
set -euo pipefail
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
J=/root/rna_jepa_data/ss_data/jsonl
D=/root/rna_jepa_data
LOG=$W/logs/build_clean_assets.log
mkdir -p "$W/logs"
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools

BLOCKS=(
  --blocklist "$J/bprna_ts0.jsonl"
  --blocklist "$J/bprna_new.jsonl"
  --blocklist "$J/ref_pdb_ts1.jsonl"
  --blocklist "$J/ref_pdb_ts2.jsonl"
  --blocklist "$J/ref_pdb_ts3.jsonl"
  --blocklist "$J/ref_pdb_ts_hard.jsonl"
  --blocklist "$J/testsetb.jsonl"
  --blocklist "$J/archiveii_embok_clean.jsonl"
  --blocklist "$J/archiveii.jsonl"
  --blocklist "$J/chanrg_test.jsonl"
  --blocklist "$J/chanrg_gena.jsonl"
  --blocklist "$J/chanrg_genc.jsonl"
  --blocklist "$J/chanrg_genf.jsonl"
  --blocklist "$J/chanrg_train_validation.jsonl"
)

check_complete() {
  "$PY" -c 'import json,sys; from pathlib import Path; j,t=Path(sys.argv[1]),Path(sys.argv[2]); m=json.loads((t/"manifest.json").read_text()); assert j.is_file(); assert m["n_sequences"] > 0; assert m["output"]["n_records"] == m["n_sequences"]; assert m["output"]["independent_second_pass"] is True; assert m["output"]["blocklist_overlap"] == 0; print("ASSET_OK", j, m["n_sequences"])' "$1" "$2"
}

if [ ! -f "$W/data/validation/chanrg_train_validation.jsonl" ]; then
  echo "[assets] $(date) validation build start" >> "$LOG"
  "$PY" tools/build_clean_corpus.py --role validation \
    --input "$J/chanrg_train_validation.jsonl" gt \
    --out-jsonl "$W/data/validation/chanrg_train_validation.jsonl" \
    --out-teacher-dir "$W/data/validation_teacher" >> "$LOG" 2>&1
fi
check_complete "$W/data/validation/chanrg_train_validation.jsonl" "$W/data/validation_teacher" >> "$LOG"

if [ ! -f "$W/data/chanrg_clean/chanrg_train_train.jsonl" ]; then
  echo "[assets] $(date) clean CHANRG build start" >> "$LOG"
  "$PY" tools/build_clean_corpus.py --role train \
    --input "$J/chanrg_train_train.jsonl" gt "$D/ss_data/teacher_gt/chanrg_train" \
    "${BLOCKS[@]}" \
    --out-jsonl "$W/data/chanrg_clean/chanrg_train_train.jsonl" \
    --out-teacher-dir "$W/data/chanrg_clean_teacher" >> "$LOG" 2>&1
fi
check_complete "$W/data/chanrg_clean/chanrg_train_train.jsonl" "$W/data/chanrg_clean_teacher" >> "$LOG"

if [ ! -f "$W/data/mixed/mixed_train.jsonl" ]; then
  echo "[assets] $(date) clean mixed build start" >> "$LOG"
  "$PY" tools/build_clean_corpus.py --role train \
    --input "$J/bprna_tr1c.jsonl" mfe "$D/ss_data/teacher/bprna_tr1c" \
    --input "$J/chanrg_train_train.jsonl" gt "$D/ss_data/teacher_gt/chanrg_train" \
    "${BLOCKS[@]}" \
    --out-jsonl "$W/data/mixed/mixed_train.jsonl" \
    --out-teacher-dir "$W/data/mixed_teacher" >> "$LOG" 2>&1
fi
check_complete "$W/data/mixed/mixed_train.jsonl" "$W/data/mixed_teacher" >> "$LOG"
sha256sum "$W/data/validation_teacher/manifest.json" \
  "$W/data/chanrg_clean_teacher/manifest.json" \
  "$W/data/mixed_teacher/manifest.json" >> "$LOG"
echo "[assets] $(date) ALL_DONE validation + clean CHANRG + clean mixed" >> "$LOG"
