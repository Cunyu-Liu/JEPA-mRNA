#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
for experiment in structrfm_frozen structrfm_full structrfm_s3; do
  while [ ! -f "$ROOT/eval_decision/$experiment/summary.json" ]; do sleep 60; done
done
$PY "$REPO/tools/compare_structrfm_paired_round48.py" --root "$ROOT" --bootstrap 10000 \
  --out "$ROOT/eval_decision/structrfm_paired_bootstrap.json" \
  > "$ROOT/logs/round48/structrfm_paired_bootstrap.log" 2>&1
$PY "$REPO/tools/compare_structrfm_paired_round48.py" --root "$ROOT" --bootstrap 10000 \
  --out "$ROOT/eval_decision/structrfm_paired_bootstrap.recheck.json" \
  >> "$ROOT/logs/round48/structrfm_paired_bootstrap.log" 2>&1
cmp "$ROOT/eval_decision/structrfm_paired_bootstrap.json" \
    "$ROOT/eval_decision/structrfm_paired_bootstrap.recheck.json"
