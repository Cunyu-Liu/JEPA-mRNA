#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
for arm in ripe_head_log ripe_head_r2 ripe_head_r3 ripe_head_r5 ripe_head_r6 ripe_encoder_r3; do
  while [[ ! -f "$ROOT/eval_decision/$arm/confidence_summary.json" ]]; do sleep 60; done
  $PY "$REPO/tools/analyze_confidence_round49.py" --root "$ROOT" --experiment "$arm" \
    --run-prefix "${arm}_s" --out "$ROOT/eval_decision/$arm/confidence_independent.json" \
    > "$ROOT/logs/round48/${arm}_confidence_independent.log" 2>&1
  $PY "$REPO/tools/analyze_confidence_round49.py" --root "$ROOT" --experiment "$arm" \
    --run-prefix "${arm}_s" --out "$ROOT/eval_decision/$arm/confidence_independent.recheck.json" \
    >> "$ROOT/logs/round48/${arm}_confidence_independent.log" 2>&1
  cmp "$ROOT/eval_decision/$arm/confidence_independent.json" "$ROOT/eval_decision/$arm/confidence_independent.recheck.json"
done
