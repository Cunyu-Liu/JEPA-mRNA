#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev:/home/rna_jev/tools
mkdir -p "$ROOT/logs/round55" "$ROOT/audits/round55"
log(){ echo "[$(date -Is)] $*" | tee -a "$ROOT/logs/round55/cl2_cl5_b0.log"; }

run_summary(){
  local arm=$1
  for suffix in json recheck.json; do
    $PY tools/qa_rinalmo_cl2_training_round54.py --arm "$arm" --summary \
      --output "$ROOT/audits/round55/${arm}_b0_training_summary.$suffix"
  done
  cmp "$ROOT/audits/round55/${arm}_b0_training_summary.json" "$ROOT/audits/round55/${arm}_b0_training_summary.recheck.json"
}

run_arm(){
  local arm=$1 seed=$2
  local run=$ROOT/runs/round54_${arm}_s${seed}
  if [[ ! -f "$run/run_meta.json" ]] || ! $PY -c 'import json,sys; assert json.load(open(sys.argv[1])).get("converged", False)' "$run/run_meta.json"; then
    log "START_OR_RESUME $arm seed=$seed"
    bash "tools/run_rinalmo_${arm}_round54.sh" "$seed" >> "$ROOT/logs/round55/${arm}_s${seed}.log" 2>&1
  fi
  for suffix in json recheck.json; do
    $PY tools/qa_rinalmo_cl2_training_round54.py --arm "$arm" --seed "$seed" \
      --output "$ROOT/audits/round55/${arm}_s${seed}_training_qa.$suffix"
  done
  cmp "$ROOT/audits/round55/${arm}_s${seed}_training_qa.json" "$ROOT/audits/round55/${arm}_s${seed}_training_qa.recheck.json"
}

log "CL2 existing three seeds: direct B0 comparison only"
run_summary cl2
for arm in cl3 cl4; do
  for seed in 42 2026 3407; do
    run_arm "$arm" "$seed"
  done
  run_summary "$arm"
done
if ! cmp -s "$ROOT/audits/round55/cl5_smoke_qa.json" "$ROOT/audits/round55/cl5_smoke_qa.recheck.json"; then
  log "START CL5 smoke after CL4 releases GPU"
  bash tools/run_rinalmo_cl5_smoke_round54.sh >> "$ROOT/logs/round55/cl5_smoke.log" 2>&1
  for suffix in json recheck.json; do
    $PY tools/qa_rinalmo_cl5_smoke_round55.py --output "$ROOT/audits/round55/cl5_smoke_qa.$suffix"
  done
  cmp "$ROOT/audits/round55/cl5_smoke_qa.json" "$ROOT/audits/round55/cl5_smoke_qa.recheck.json"
fi
for seed in 42 2026 3407; do
  run_arm cl5 "$seed"
done
run_summary cl5
log "CL2_CL5_B0_TRAINING_COMPLETE"
