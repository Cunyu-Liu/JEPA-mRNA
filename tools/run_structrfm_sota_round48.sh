#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
SELECTION=${1:-all}
case "$SELECTION" in
  all) experiments=(structrfm_frozen structrfm_full structrfm_s3) ;;
  structrfm_frozen|structrfm_full|structrfm_s3) experiments=("$SELECTION") ;;
  *) echo "unknown experiment selection: $SELECTION" >&2; exit 2 ;;
esac
for experiment in "${experiments[@]}"; do
  while [ ! -f "$ROOT/eval_decision/$experiment/summary.json" ]; do sleep 60; done
  $PY "$REPO/tools/compare_structrfm_sota_round48.py" --root "$ROOT" --experiment "$experiment" \
    --baseline "$ROOT/source_data/round48/round39_own_clean_leaderboard.json" \
    --rfam-baseline "$ROOT/source_data/round48/round43_bpfold_rfam.json" \
    --chanrg-baseline "$ROOT/source_data/round48/round40_baseline_ood_summary.json" \
    > "$ROOT/logs/round48/${experiment}_sota_comparison.log" 2>&1
done
