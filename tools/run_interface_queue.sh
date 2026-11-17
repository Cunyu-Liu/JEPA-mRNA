#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
cd "$REPO"
for MODE in sequence_only structure_aware length gc random; do
  for SEED in 42 2026 3407; do
    if [ -f "$ART/artifacts/v2/interface/${MODE}_seed${SEED}/result.json" ]; then
      continue
    fi
    "$PY" tools/train_interface_baseline.py --mode "$MODE" --seed "$SEED" --max-steps 500 > "$ART/artifacts/v2/interface/${MODE}_seed${SEED}.stdout.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do
      sleep 5
    done
  done
  wait
done
"$PY" tools/aggregate_interface_results.py
