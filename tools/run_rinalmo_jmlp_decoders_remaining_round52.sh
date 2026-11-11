#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
cd "$REPO"
until [[ $(find "$ROOT/evals/round52/jmlp_greedy" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l) -ge 16 ]]; do
  sleep 300
done
for seed in 42 2026 3407; do
  bash tools/run_rinalmo_jmlp_benchmark_round52.sh large "$seed" greedy \
    > "$ROOT/logs/round52/rinalmo_jmlp_greedy_large_s${seed}_queue.log" 2>&1
done
for seed in 42 2026 3407; do
  bash tools/run_rinalmo_jmlp_benchmark_round52.sh small "$seed" turner0 \
    > "$ROOT/logs/round52/rinalmo_jmlp_turner0_small_s${seed}_queue.log" 2>&1 &
done
wait
for seed in 42 2026 3407; do
  bash tools/run_rinalmo_jmlp_benchmark_round52.sh large "$seed" turner0 \
    > "$ROOT/logs/round52/rinalmo_jmlp_turner0_large_s${seed}_queue.log" 2>&1
done
for mode in greedy turner0; do
  "$PY" tools/qa_rinalmo_jmlp_decoders_round52.py --root "$ROOT" --mode "$mode" \
    --output "$ROOT/audits/round52/rinalmo_jmlp_${mode}_benchmark_qa.json" \
    > "$ROOT/logs/round52/rinalmo_jmlp_${mode}_benchmark_qa.log" 2>&1
  "$PY" tools/qa_rinalmo_jmlp_decoders_round52.py --root "$ROOT" --mode "$mode" \
    --output "$ROOT/audits/round52/rinalmo_jmlp_${mode}_benchmark_qa.recheck.json" \
    > "$ROOT/logs/round52/rinalmo_jmlp_${mode}_benchmark_qa_recheck.log" 2>&1
  cmp "$ROOT/audits/round52/rinalmo_jmlp_${mode}_benchmark_qa.json" \
      "$ROOT/audits/round52/rinalmo_jmlp_${mode}_benchmark_qa.recheck.json"
done
