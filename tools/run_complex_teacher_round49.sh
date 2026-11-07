#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
DATA="$ROOT/data/rna_complex_round49"
PY=/root/miniconda3/envs/lucaone/bin/python
while [[ ! -f "$DATA/decontamination/verification.json" ]]; do sleep 60; done
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT" and d["n_train"]>0' "$DATA/decontamination/verification.json"
cd /home/rna_jev
$PY tools/parallel_teacher_labels.py --data "$DATA/decontamination/complex_train.jsonl" \
  --out "$DATA/teacher_complex" --workers 8 --shard-size 256 --max-length 0 \
  --python "$PY" > "$ROOT/logs/round48/rna_complex_teacher.log" 2>&1
while [[ ! -f "$ROOT/audits/round48/pairjepa_decision.json" ]]; do sleep 60; done
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["allow_complex_data_route"]' "$ROOT/audits/round48/pairjepa_decision.json"
$PY tools/prepare_complex_training_round49.py \
  --decontamination "$DATA/decontamination" --teacher-complex "$DATA/teacher_complex" \
  --tr1c /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-tr1c /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --decision "$ROOT/audits/round48/pairjepa_decision.json" \
  --out "$DATA/d1_preparation" > "$ROOT/logs/round48/rna_complex_d1_preparation.log" 2>&1
