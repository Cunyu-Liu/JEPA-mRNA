#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
while [[ ! -f "$ROOT/eval_decision/speed_round49/spotrna_clean_ts0.json" ]]; do sleep 60; done
$PY "$REPO/tools/summarize_speed_round49.py" --root "$ROOT" \
  --out "$ROOT/eval_decision/speed_round49/summary.json" > "$ROOT/logs/round48/speed_summary.log" 2>&1
$PY "$REPO/tools/summarize_speed_round49.py" --root "$ROOT" \
  --out "$ROOT/eval_decision/speed_round49/summary.recheck.json" >> "$ROOT/logs/round48/speed_summary.log" 2>&1
cmp "$ROOT/eval_decision/speed_round49/summary.json" "$ROOT/eval_decision/speed_round49/summary.recheck.json"
