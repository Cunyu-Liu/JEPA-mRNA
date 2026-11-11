#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
exec 9>/tmp/round52_rinalmo_jmlp_benchmark_qa.lock
flock -n 9 || { echo "another J-mlp benchmark QA is active"; exit 1; }
until [[ $(find "$ROOT/evals/round52/jmlp_benchmark" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l) -eq 27 ]]; do
  sleep 300
done
"$PY" "$REPO/tools/qa_rinalmo_jmlp_benchmark_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_jmlp_benchmark_qa.json" \
  > "$ROOT/logs/round52/rinalmo_jmlp_benchmark_qa.log" 2>&1
"$PY" "$REPO/tools/qa_rinalmo_jmlp_benchmark_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_jmlp_benchmark_qa.recheck.json" \
  > "$ROOT/logs/round52/rinalmo_jmlp_benchmark_qa_recheck.log" 2>&1
cmp "$ROOT/audits/round52/rinalmo_jmlp_benchmark_qa.json" \
    "$ROOT/audits/round52/rinalmo_jmlp_benchmark_qa.recheck.json"
