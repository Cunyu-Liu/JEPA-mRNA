#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
while [[ ! -f "$ROOT/eval_decision/complex_d1_greedy/summary.json" || ! -f "$ROOT/eval_decision/complex_d1_turner0/summary.json" ]]; do sleep 60; done
cd /home/rna_jev
$PY tools/compare_ripe_paired_round48.py --root "$ROOT" \
  --manifest data/rna_complex_round49/d1_preparation/comparison_manifest.json \
  --status PASS_COMPLEX_D1_PAIRED_MICRO_F1_BOOTSTRAP \
  --out "$ROOT/eval_decision/complex_d1_paired_bootstrap.json" > "$ROOT/logs/round48/complex_d1_paired.log" 2>&1
$PY tools/adjudicate_complex_d1_round49.py --root "$ROOT" \
  --out "$ROOT/audits/round48/complex_d1_decision.json" > "$ROOT/logs/round48/complex_d1_decision.log" 2>&1
$PY tools/adjudicate_complex_d1_round49.py --root "$ROOT" \
  --out "$ROOT/audits/round48/complex_d1_decision.recheck.json" >> "$ROOT/logs/round48/complex_d1_decision.log" 2>&1
cmp "$ROOT/audits/round48/complex_d1_decision.json" "$ROOT/audits/round48/complex_d1_decision.recheck.json"
