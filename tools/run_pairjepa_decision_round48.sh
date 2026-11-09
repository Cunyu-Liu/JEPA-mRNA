#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
while [ ! -f "$ROOT/audits/round48/pairjepa_preregistered.json" ]; do sleep 60; done
readarray -t ARMS < <($PY -c 'import json; d=json.load(open("'"$ROOT"'/audits/round48/pairjepa_preregistered.json")); [print(a["arm"]) for a in d["arms"] if not a["reuse_baseline"]]')
for arm in "${ARMS[@]}"; do
  for relative in summary.json sota_comparison.json confidence_summary.json; do
    while [ ! -f "$ROOT/eval_decision/$arm/$relative" ]; do sleep 60; done
  done
  for ablation in greedy turner0; do
    while [ ! -f "$ROOT/eval_decision/${arm}_${ablation}/summary.json" ]; do sleep 60; done
  done
  for suffix in "" "_greedy" "_turner0"; do
    $PY "$REPO/tools/qa_pairjepa_round49.py" --root "$ROOT" --experiment "${arm}${suffix}" \
      --out "$ROOT/audits/round49/${arm}${suffix}_independent_qa.json" \
      > "$ROOT/logs/round48/${arm}${suffix}_independent_qa.log" 2>&1
    $PY "$REPO/tools/qa_pairjepa_round49.py" --root "$ROOT" --experiment "${arm}${suffix}" \
      --out "$ROOT/audits/round49/${arm}${suffix}_independent_qa.recheck.json" \
      >> "$ROOT/logs/round48/${arm}${suffix}_independent_qa.log" 2>&1
    cmp "$ROOT/audits/round49/${arm}${suffix}_independent_qa.json" \
      "$ROOT/audits/round49/${arm}${suffix}_independent_qa.recheck.json"
  done
done
$PY "$REPO/tools/compare_ripe_paired_round48.py" --root "$ROOT" --bootstrap 10000 \
  --manifest audits/round48/pairjepa_preregistered.json --status PASS_PAIRJEPA_PAIRED_MICRO_F1_BOOTSTRAP \
  --out "$ROOT/eval_decision/pairjepa_paired_bootstrap.json" > "$ROOT/logs/round48/pairjepa_paired.log" 2>&1
$PY "$REPO/tools/compare_ripe_paired_round48.py" --root "$ROOT" --bootstrap 10000 \
  --manifest audits/round48/pairjepa_preregistered.json --status PASS_PAIRJEPA_PAIRED_MICRO_F1_BOOTSTRAP \
  --out "$ROOT/eval_decision/pairjepa_paired_bootstrap.recheck.json" >> "$ROOT/logs/round48/pairjepa_paired.log" 2>&1
cmp "$ROOT/eval_decision/pairjepa_paired_bootstrap.json" "$ROOT/eval_decision/pairjepa_paired_bootstrap.recheck.json"
$PY "$REPO/tools/adjudicate_pairjepa_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/pairjepa_decision.json" \
  > "$ROOT/logs/round48/pairjepa_decision.log" 2>&1
$PY "$REPO/tools/adjudicate_pairjepa_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/pairjepa_decision.recheck.json" \
  >> "$ROOT/logs/round48/pairjepa_decision.log" 2>&1
cmp "$ROOT/audits/round48/pairjepa_decision.json" "$ROOT/audits/round48/pairjepa_decision.recheck.json"
