#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
required=(eval_decision/structrfm_paired_bootstrap.json)
for base in structrfm_frozen structrfm_full structrfm_s3; do
  required+=(eval_decision/$base/summary.json eval_decision/$base/sota_comparison.json eval_decision/$base/confidence_summary.json)
  required+=(eval_decision/${base}_greedy/summary.json eval_decision/${base}_turner0/summary.json)
done
for relative in "${required[@]}"; do
  while [ ! -f "$ROOT/$relative" ]; do sleep 60; done
done
$PY "$REPO/tools/adjudicate_structrfm_route1_round48.py" --root "$ROOT" \
  --out "$ROOT/audits/round48/route1_decision.json" \
  > "$ROOT/logs/round48/route1_decision.log" 2>&1
$PY "$REPO/tools/adjudicate_structrfm_route1_round48.py" --root "$ROOT" \
  --out "$ROOT/audits/round48/route1_decision.recheck.json" \
  >> "$ROOT/logs/round48/route1_decision.log" 2>&1
cmp "$ROOT/audits/round48/route1_decision.json" "$ROOT/audits/round48/route1_decision.recheck.json"
