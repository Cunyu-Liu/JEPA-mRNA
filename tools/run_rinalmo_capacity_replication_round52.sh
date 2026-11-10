#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_capacity_replication.lock
flock -n 9 || { echo "another capacity replication queue is active"; exit 1; }
until test -f "$ROOT/audits/round52/rinalmo_capacity_diagnostics_complete.json"; do
  sleep 300
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
while (( used > 14000 )); do
  sleep 300
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
done
for arm in length olr; do
  checkpoint="$ROOT/runs/round52_rinalmo_${arm}_s42/best.pt"
  error="$ROOT/audits/round52/rinalmo_capacity_error_${arm}_s42.repeat.json"
  representation="$ROOT/audits/round52/rinalmo_capacity_representation_${arm}_s42.repeat.json"
  if [[ -e "$error" || -e "$representation" ]]; then
    echo "refusing to overwrite capacity replication for $arm" >&2
    exit 1
  fi
  "$PY" "$REPO/tools/probe_rinalmo_capacity_error_round52.py" \
    --checkpoint "$checkpoint" \
    --train-data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --train-examples 196 --seed 42 --output "$error" \
    > "$ROOT/logs/round52/rinalmo_capacity_error_${arm}_s42.repeat.log" 2>&1
  "$PY" "$REPO/tools/probe_rinalmo_frozen_representation_round52.py" \
    --checkpoint "$checkpoint" \
    --train-data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --samples-per-sequence 256 --seed 42 --output "$representation" \
    > "$ROOT/logs/round52/rinalmo_capacity_representation_${arm}_s42.repeat.log" 2>&1
done
"$PY" "$REPO/tools/qa_rinalmo_capacity_replication_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_capacity_replication.json"
"$PY" "$REPO/tools/qa_rinalmo_capacity_replication_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_capacity_replication.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_capacity_replication.json" \
    "$ROOT/audits/round52/rinalmo_capacity_replication.recheck.json"
"$PY" "$REPO/tools/decide_rinalmo_capacity_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_capacity_decision.json"
"$PY" "$REPO/tools/decide_rinalmo_capacity_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_capacity_decision.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_capacity_decision.json" \
    "$ROOT/audits/round52/rinalmo_capacity_decision.recheck.json"
