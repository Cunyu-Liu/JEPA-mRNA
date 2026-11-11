#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
exec 9>/tmp/round52_rinalmo_jmlp.lock
flock -n 9 || { echo "another J-mlp QA is active"; exit 1; }
until test -f "$ROOT/audits/round52/rinalmo_capacity_decision.json"; do
  sleep 300
done
"$PY" "$REPO/tools/qa_rinalmo_jmlp_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_jmlp_training_qa.json"
"$PY" "$REPO/tools/qa_rinalmo_jmlp_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_jmlp_training_qa.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_jmlp_training_qa.json" \
    "$ROOT/audits/round52/rinalmo_jmlp_training_qa.recheck.json"
