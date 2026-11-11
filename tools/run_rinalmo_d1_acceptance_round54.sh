#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev:/home/rna_jev/tools
while kill -0 475241 2>/dev/null; do sleep 30; done
$PY - <<'PY'
import json
from pathlib import Path
p=Path('/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/runs/round54_b0_s3407/run_meta.json')
d=json.loads(p.read_text())
assert d['status'] == 'completed' and d['converged'] and d['stop_reason'] == 'early_stopping', d
PY
if ! cmp -s "$ROOT/audits/round54/b0_training_qa.json" "$ROOT/audits/round54/b0_training_qa.recheck.json"; then
  for suffix in json recheck.json; do
    $PY tools/qa_rinalmo_b0_training_round54.py --root "$ROOT" --output "$ROOT/audits/round54/b0_training_qa.$suffix"
  done
  cmp "$ROOT/audits/round54/b0_training_qa.json" "$ROOT/audits/round54/b0_training_qa.recheck.json"
fi
evaluate_holdout() {
  local arm=$1 seed=$2
  local out=$ROOT/evals/round54/${arm}_s${seed}_complex_holdout_1022
  local qa=$ROOT/audits/round54/${arm}_s${seed}_holdout_qa
  if cmp -s "$qa.json" "$qa.recheck.json"; then return; fi
  if [[ ! -f "$out/result.json" ]]; then
    mkdir -p "$out"
    $PY tools/eval_rinalmo_full_round52.py \
      --checkpoint "$ROOT/runs/round54_${arm}_s${seed}/best.pt" \
      --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
      --data "$ROOT/data/rna_complex_round49/rinalmo_d1_length_round54/1022/complex_holdout.jsonl" \
      --out "$out/result.json" --repeats 3 --decoder nussinov --dp-backend c \
      > "$out/eval.log" 2>&1
  fi
  for suffix in json recheck.json; do
    $PY tools/qa_rinalmo_d1_round54.py --mode holdout --arm "$arm" --seed "$seed" \
      --output "$qa.$suffix"
  done
  cmp "$qa.json" "$qa.recheck.json"
}
for seed in 42 2026 3407; do
  evaluate_holdout b0 "$seed"
done
for seed in 42 2026 3407; do
  if [[ ! -f "$ROOT/runs/round54_d1_s${seed}/run_meta.json" ]]; then
    bash tools/run_rinalmo_d1_round54.sh "$seed" > "$ROOT/logs/round54/d1_s${seed}.log" 2>&1
  fi
  if ! cmp -s "$ROOT/audits/round54/d1_s${seed}_training_qa.json" "$ROOT/audits/round54/d1_s${seed}_training_qa.recheck.json"; then
    for suffix in json recheck.json; do
      $PY tools/qa_rinalmo_d1_round54.py --mode training --seed "$seed" --output "$ROOT/audits/round54/d1_s${seed}_training_qa.$suffix"
    done
    cmp "$ROOT/audits/round54/d1_s${seed}_training_qa.json" "$ROOT/audits/round54/d1_s${seed}_training_qa.recheck.json"
  fi
  evaluate_holdout d1 "$seed"
done
for suffix in json recheck.json; do
  $PY tools/summarize_rinalmo_d1_round54.py --output "$ROOT/audits/round54/d0_d1_summary_qa.$suffix"
done
cmp "$ROOT/audits/round54/d0_d1_summary_qa.json" "$ROOT/audits/round54/d0_d1_summary_qa.recheck.json"
