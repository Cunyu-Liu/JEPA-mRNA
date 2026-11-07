#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
for arm in ripe_head_log ripe_head_r1 ripe_head_r2 ripe_head_r3 ripe_head_r5 ripe_head_r6 ripe_encoder_r3; do
  for relative in summary.json sota_comparison.json confidence_summary.json; do
    while [ ! -f "$ROOT/eval_decision/$arm/$relative" ]; do sleep 60; done
  done
  for ablation in greedy turner0; do
    while [ ! -f "$ROOT/eval_decision/${arm}_${ablation}/summary.json" ]; do sleep 60; done
  done
done
$PY "$REPO/tools/compare_ripe_paired_round48.py" --root "$ROOT" --bootstrap 10000 \
  --out "$ROOT/eval_decision/ripe_paired_bootstrap.json" > "$ROOT/logs/round48/ripe_paired.log" 2>&1
$PY "$REPO/tools/compare_ripe_paired_round48.py" --root "$ROOT" --bootstrap 10000 \
  --out "$ROOT/eval_decision/ripe_paired_bootstrap.recheck.json" >> "$ROOT/logs/round48/ripe_paired.log" 2>&1
cmp "$ROOT/eval_decision/ripe_paired_bootstrap.json" "$ROOT/eval_decision/ripe_paired_bootstrap.recheck.json"
$PY "$REPO/tools/adjudicate_ripe_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/ripe_decision.json" \
  > "$ROOT/logs/round48/ripe_decision.log" 2>&1
$PY "$REPO/tools/adjudicate_ripe_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/ripe_decision.recheck.json" \
  >> "$ROOT/logs/round48/ripe_decision.log" 2>&1
cmp "$ROOT/audits/round48/ripe_decision.json" "$ROOT/audits/round48/ripe_decision.recheck.json"
