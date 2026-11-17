#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
ROOT="$ART/artifacts/p2/pilot_v1"
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
cd "$REPO"
run_eval() {
  ARM="$1"
  if [ -f "$ROOT/$ARM/model_dev_eval.json" ]; then
    return 0
  fi
  "$PY" tools/evaluate_single_checkpoint.py \
    --checkpoint "$ROOT/$ARM/best.pt" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
    --data-role model-dev --repeats 1 \
    --output "$ROOT/$ARM/model_dev_eval.json" \
    >> "$ROOT/$ARM/model_dev_eval.stdout.log" 2>&1
}
for ARM in H0 H1; do run_eval "$ARM" & done
wait
for ARM in H2 H3; do run_eval "$ARM" & done
wait
run_eval H4
"$PY" tools/aggregate_dual_view_pilot.py --artifact-root "$ART" --output "$ART/manifests/p2/pilot_v1_decision.json"
