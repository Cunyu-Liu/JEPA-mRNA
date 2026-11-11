#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
bash tools/run_rinalmo_jmlp_benchmark_round52.sh large 2026 \
  > /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/logs/round52/rinalmo_jmlp_benchmark_large_s2026_queue.log 2>&1
bash tools/run_rinalmo_jmlp_benchmark_round52.sh large 3407 \
  > /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/logs/round52/rinalmo_jmlp_benchmark_large_s3407_queue.log 2>&1
bash tools/run_rinalmo_jmlp_benchmark_qa_round52.sh \
  > /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/logs/round52/rinalmo_jmlp_benchmark_qa_queue.log 2>&1
