#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round51_capacity_error.lock
flock -n 9 || { echo "another capacity error probe is active"; exit 1; }
pids=()
label=${1:-v1}
[[ "$label" =~ ^[a-z0-9]+$ ]] || { echo "invalid artifact label"; exit 1; }
for seed in 42 2026 3407; do
  output="$ROOT/audits/round51/capacity_error_${label}_s${seed}.json"
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite $output" >&2
    exit 1
  fi
  "$PY" "$REPO/tools/probe_capacity_error_round51.py" \
    --checkpoint "$ROOT/runs/loss_length_s${seed}/best.pt" \
    --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --train-data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --output "$output" --train-examples 196 --seed 42 \
    >> "$ROOT/logs/round51/capacity_error_${label}_s${seed}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
