#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round52_rinalmo_long_sequence_remaining.lock
flock -n 9 || { echo "another long-sequence queue is active"; exit 1; }
for seed in 2026 3407; do
  out="$ROOT/audits/round52/rinalmo_long_sequence_probe_s${seed}.json"
  if [[ -e "$out" ]]; then echo "refusing to overwrite $out" >&2; exit 1; fi
  "$PY" "$REPO/tools/probe_rinalmo_long_sequence_round52.py" \
    --checkpoint "$ROOT/runs/round52_rinalmo_length_s${seed}/best.pt" \
    --lengths 512 1022 1024 1500 1869 3819 --output "$out" \
    > "$ROOT/logs/round52/rinalmo_long_sequence_probe_s${seed}.log" 2>&1
done
"$PY" "$REPO/tools/qa_rinalmo_long_sequence_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_long_sequence_remaining_qa.json"
"$PY" "$REPO/tools/qa_rinalmo_long_sequence_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_long_sequence_remaining_qa.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_long_sequence_remaining_qa.json" \
    "$ROOT/audits/round52/rinalmo_long_sequence_remaining_qa.recheck.json"
