#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev:/home/rna_jev/tools
mkdir -p "$ROOT/logs/round54" "$ROOT/audits/round54"
log(){ echo "[$(date -Is)] $*" | tee -a "$ROOT/logs/round54/prism_remaining.log"; }
decision(){ printf '%s\n' "$1" | tee -a "$ROOT/audits/round54/pipeline_decisions.jsonl"; }
wait_pid(){ local pid=$1; while kill -0 "$pid" 2>/dev/null; do sleep 60; done; }
mean_vl0(){ $PY -c "import json; print(json.load(open('$1'))['mean_best_vl0'])"; }
run_arm(){ local arm=$1 seed=$2
  local run=$ROOT/runs/round54_${arm}_s${seed}
  local qa=$ROOT/audits/round54/${arm}_s${seed}_training_qa
  if [[ ! -f "$run/run_meta.json" ]]; then
    log "START $arm seed=$seed"
    bash "tools/run_rinalmo_${arm}_round54.sh" "$seed" > "$ROOT/logs/round54/${arm}_s${seed}.log" 2>&1
  fi
  if ! cmp -s "$qa.json" "$qa.recheck.json"; then
    for suffix in json recheck.json; do
      case "$arm" in
        cl2|cl3|cl4) $PY tools/qa_rinalmo_cl2_training_round54.py --arm "$arm" --seed "$seed" --output "$qa.$suffix" ;;
        e2a_motif|e2a_shuffled) $PY tools/qa_rinalmo_e2a_round54.py --arm "$arm" --seed "$seed" --output "$qa.$suffix" ;;
        e2b) $PY tools/qa_rinalmo_e2b_round54.py --seed "$seed" --output "$qa.$suffix" ;;
        e2c) $PY tools/qa_rinalmo_e2c_round54.py --seed "$seed" --output "$qa.$suffix" ;;
        *) log "UNKNOWN_ARM $arm"; return 1 ;;
      esac
    done
    cmp "$qa.json" "$qa.recheck.json"
  fi
}
run_summary(){ local arm=$1
  if ! cmp -s "$ROOT/audits/round54/${arm}_training_summary.json" "$ROOT/audits/round54/${arm}_training_summary.recheck.json"; then
    for suffix in json recheck.json; do
      case "$arm" in
        cl2|cl3|cl4) $PY tools/qa_rinalmo_cl2_training_round54.py --arm "$arm" --summary --output "$ROOT/audits/round54/${arm}_training_summary.$suffix" ;;
        e2a_motif|e2a_shuffled) $PY tools/qa_rinalmo_e2a_round54.py --arm "$arm" --summary --output "$ROOT/audits/round54/${arm}_training_summary.$suffix" ;;
        e2b) $PY tools/qa_rinalmo_e2b_round54.py --summary --output "$ROOT/audits/round54/${arm}_training_summary.$suffix" ;;
        e2c) $PY tools/qa_rinalmo_e2c_round54.py --summary --output "$ROOT/audits/round54/${arm}_training_summary.$suffix" ;;
        *) log "UNKNOWN_ARM $arm"; return 1 ;;
      esac
    done
    cmp "$ROOT/audits/round54/${arm}_training_summary.json" "$ROOT/audits/round54/${arm}_training_summary.recheck.json"
  fi
}
run_seeds(){ local arm=$1; for seed in 42 2026 3407; do run_arm "$arm" "$seed"; done; run_summary "$arm"; }
require_tools(){ for file in "$@"; do [[ -f "$file" ]] || { log "MISSING_REQUIRED_TOOL $file"; return 1; }; done; }
log "WAIT existing CL2 queue PID=613548"
wait_pid 613548
if ! cmp -s "$ROOT/audits/round54/cl2_training_summary.json" "$ROOT/audits/round54/cl2_training_summary.recheck.json"; then
  bash tools/run_rinalmo_cl2_acceptance_round54.sh >> "$ROOT/logs/round54/cl2_acceptance_r5.log" 2>&1
fi
cl2_mean=$(mean_vl0 "$ROOT/audits/round54/cl2_training_summary.json")
cl1_mean=$(mean_vl0 "$ROOT/audits/round54/cl1_training_summary.json")
if ! $PY -c "assert $cl2_mean < $cl1_mean"; then
  decision '{"stage":"E1","decision":"STOP_CL2_NOT_BETTER_THAN_CL1"}'
else
  if ! cmp -s "$ROOT/audits/round54/cl3_smoke_qa.json" "$ROOT/audits/round54/cl3_smoke_qa.recheck.json"; then
    log "CL3 smoke after GPU release"
    bash tools/run_rinalmo_cl3_smoke_round54.sh > "$ROOT/logs/round54/cl3_smoke_s42_r3.log" 2>&1
    for suffix in json recheck.json; do
      $PY tools/qa_rinalmo_cl2_smoke_round54.py --expected-arm cl3 --run "$ROOT/runs/round54_cl3_smoke_s42" --output "$ROOT/audits/round54/cl3_smoke_qa.$suffix"
    done
    cmp "$ROOT/audits/round54/cl3_smoke_qa.json" "$ROOT/audits/round54/cl3_smoke_qa.recheck.json"
  fi
  run_seeds cl3
  cl3_mean=$(mean_vl0 "$ROOT/audits/round54/cl3_training_summary.json")
  if ! $PY -c "assert $cl3_mean < $cl2_mean"; then
    decision '{"stage":"E1","decision":"STOP_CL3_NOT_BETTER_THAN_CL2"}'
  else
    run_seeds cl4
    cl4_mean=$(mean_vl0 "$ROOT/audits/round54/cl4_training_summary.json")
    if ! $PY -c "assert $cl4_mean < $cl3_mean"; then
      decision '{"stage":"E1","decision":"STOP_CL4_NOT_BETTER_THAN_CL3"}'
    else
      require_tools tools/run_rinalmo_cl5_round54.sh
      run_seeds cl5
    fi
  fi
fi
log "START independent E2a motif arm"
require_tools tools/run_rinalmo_e2a_motif_round54.sh tools/qa_rinalmo_e2a_round54.py
run_seeds e2a_motif
require_tools tools/run_rinalmo_e2a_shuffled_round54.sh
run_seeds e2a_shuffled
log "START conditional E2b statistical thermodynamics"
require_tools tools/run_rinalmo_e2b_round54.sh tools/qa_rinalmo_e2b_round54.py
run_seeds e2b
log "START independent E2c enhanced DP"
require_tools tools/run_rinalmo_e2c_round54.sh tools/qa_rinalmo_e2c_round54.py
run_seeds e2c
log "START frozen main-eight and Rfam acceptance"
require_tools tools/run_prism_frozen_benchmark_round54.sh tools/qa_prism_frozen_benchmark_round54.py
bash tools/run_prism_frozen_benchmark_round54.sh >> "$ROOT/logs/round54/frozen_benchmark.log" 2>&1
$PY tools/qa_prism_frozen_benchmark_round54.py --root "$ROOT" --output "$ROOT/audits/round54/frozen_benchmark_qa.json"
log "ALL_REGISTERED_STAGES_COMPLETE"
