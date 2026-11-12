#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev:/home/rna_jev/tools
while kill -0 566946 2>/dev/null; do sleep 30; done
for seed in 42 2026 3407; do
  if [[ ! -f "$ROOT/runs/round54_cl1_s${seed}/run_meta.json" ]]; then
    bash tools/run_rinalmo_cl1_round54.sh "$seed" > "$ROOT/logs/round54/cl1_s${seed}.log" 2>&1
  fi
  for suffix in json recheck.json; do
    $PY tools/qa_rinalmo_cl1_training_round54.py --seed "$seed" --output "$ROOT/audits/round54/cl1_s${seed}_training_qa.$suffix"
  done
  cmp "$ROOT/audits/round54/cl1_s${seed}_training_qa.json" "$ROOT/audits/round54/cl1_s${seed}_training_qa.recheck.json"
done
for suffix in json recheck.json; do
  $PY tools/qa_rinalmo_cl1_training_round54.py --summary --output "$ROOT/audits/round54/cl1_training_summary.$suffix"
done
cmp "$ROOT/audits/round54/cl1_training_summary.json" "$ROOT/audits/round54/cl1_training_summary.recheck.json"
