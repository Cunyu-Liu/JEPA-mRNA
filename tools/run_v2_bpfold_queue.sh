#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
cd "$REPO"
run_pair() {
  for ARM in "$@"; do
    bash tools/run_v2_bpfold_pilot.sh "$ARM" > "$ART/artifacts/v2/bpfold_v2_${ARM}.stdout.log" 2>&1 &
  done
  wait
}
run_pair inner_outer shuffled
run_pair turner0 energy_only
"$PY" tools/aggregate_v2_bpfold.py
