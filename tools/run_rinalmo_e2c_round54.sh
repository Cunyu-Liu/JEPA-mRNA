#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools:/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/bin
for seed in 42 2026 3407; do
  if [[ ! -f "$ROOT/audits/round54/e2c_s${seed}_vl0.json" ]]; then
    /root/miniconda3/envs/lucaone/bin/python tools/evaluate_rinalmo_e2c_round54.py --seed "$seed" >> "$ROOT/logs/round54/e2c_s${seed}.log" 2>&1
  fi
done
