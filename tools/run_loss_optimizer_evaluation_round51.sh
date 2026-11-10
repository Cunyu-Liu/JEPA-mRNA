#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
export ROOT
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round51_loss_optimizer_evaluation.lock
flock -n 9 || { echo "another L/O evaluation queue is active"; exit 1; }
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS_LOSS_OPTIMIZER_TRAINING_THREE_CHECK_QA"' "$ROOT/audits/round51/loss_optimizer_training_qa.json"
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
SEEDS=(42 2026 3407)
mkdir -p "$ROOT/logs/round51" "$ROOT/audits/round51"

wait_three() {
  local fail=0 status
  set +e
  for pid in "$@"; do wait "$pid"; status=$?; if [ "$status" -ne 0 ]; then fail=1; fi; done
  set -e
  return "$fail"
}

eval_seed() {
  local arm=$1 seed=$2
  local checkpoint=$ROOT/runs/loss_${arm}_s${seed}/best.pt
  for split in "${SPLITS[@]}"; do
    local out=$ROOT/eval_decision/loss_${arm}/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["score_forward_count_per_sequence"]==1' "$out" "$checkpoint" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 > "$ROOT/logs/round51/eval_loss_${arm}_s${seed}_${split}.log" 2>&1
  done
  $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
    --pretrained "$ROOT/weights/structrfm_round48" --data "$ROOT/data/validation/vl0_196.jsonl" \
    --out "$ROOT/eval_decision/loss_${arm}/vl0_s${seed}.json" --repeats 3 \
    > "$ROOT/logs/round51/eval_loss_${arm}_s${seed}_vl0.log" 2>&1
}

ablation_seed() {
  local arm=$1 ablation=$2 seed=$3
  local checkpoint=$ROOT/runs/loss_${arm}_s${seed}/best.pt
  local experiment=loss_${arm}_${ablation}
  local extra=(--decoder greedy) expected_decoder=greedy expected_prior=none
  if [ "$ablation" = turner0 ]; then extra=(--prior-weight-override 0); expected_decoder=nussinov; expected_prior=zero; fi
  for split in "${SPLITS[@]}"; do
    local out=$ROOT/eval_decision/$experiment/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["decoder"]==sys.argv[3] and d["prior_weight_override"]==(None if sys.argv[4]=="none" else 0.0)' "$out" "$checkpoint" "$expected_decoder" "$expected_prior" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 "${extra[@]}" \
      > "$ROOT/logs/round51/eval_${experiment}_s${seed}_${split}.log" 2>&1
  done
}

for ARM in length olr; do
  pids=()
  for seed in "${SEEDS[@]}"; do eval_seed "$ARM" "$seed" & pids+=("$!"); done
  wait_three "${pids[@]}"
  $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
    --experiment "loss_${ARM}" --run-prefix "loss_${ARM}_s" --seeds "${SEEDS[@]}" \
    --out "$ROOT/eval_decision/loss_${ARM}/summary.json" > "$ROOT/logs/round51/loss_${ARM}_aggregate.log" 2>&1
  $PY "$REPO/tools/analyze_structrfm_confidence_round48.py" --root "$ROOT" --full-model \
    --experiment "loss_${ARM}" --run-prefix "loss_${ARM}_s" --seeds "${SEEDS[@]}" \
    > "$ROOT/logs/round51/loss_${ARM}_confidence.log" 2>&1
  $PY "$REPO/tools/compare_structrfm_sota_round48.py" --root "$ROOT" --experiment "loss_${ARM}" \
    --baseline "$ROOT/source_data/round48/round39_own_clean_leaderboard.json" \
    --rfam-baseline "$ROOT/source_data/round48/round43_bpfold_rfam.json" \
    --chanrg-baseline "$ROOT/source_data/round48/round40_baseline_ood_summary.json" \
    > "$ROOT/logs/round51/loss_${ARM}_sota.log" 2>&1
  for ABLATION in greedy turner0; do
    pids=()
    for seed in "${SEEDS[@]}"; do ablation_seed "$ARM" "$ABLATION" "$seed" & pids+=("$!"); done
    wait_three "${pids[@]}"
    $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
      --experiment "loss_${ARM}_${ABLATION}" --run-prefix "loss_${ARM}_s" --seeds "${SEEDS[@]}" \
      --out "$ROOT/eval_decision/loss_${ARM}_${ABLATION}/summary.json" \
      > "$ROOT/logs/round51/loss_${ARM}_${ABLATION}_aggregate.log" 2>&1
  done
  for SUFFIX in "" _greedy _turner0; do
    EXPERIMENT=loss_${ARM}${SUFFIX}
    $PY "$REPO/tools/qa_pairjepa_round49.py" --root "$ROOT" --experiment "$EXPERIMENT" \
      --out "$ROOT/audits/round51/${EXPERIMENT}_qa.json" > "$ROOT/logs/round51/${EXPERIMENT}_qa.log" 2>&1
    $PY "$REPO/tools/qa_pairjepa_round49.py" --root "$ROOT" --experiment "$EXPERIMENT" \
      --out "$ROOT/audits/round51/${EXPERIMENT}_qa.recheck.json" > "$ROOT/logs/round51/${EXPERIMENT}_qa_recheck.log" 2>&1
    cmp "$ROOT/audits/round51/${EXPERIMENT}_qa.json" "$ROOT/audits/round51/${EXPERIMENT}_qa.recheck.json"
  done
done
$PY - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ["ROOT"])
files={}
for arm in ("length","olr"):
    for suffix in ("","_greedy","_turner0"):
        experiment=f"loss_{arm}{suffix}"
        qa=root/"audits/round51"/f"{experiment}_qa.json"
        report=json.loads(qa.read_text())
        assert report["status"]=="PASS_PAIRJEPA_INDEPENDENT_QA" and report["complete"] and report["n_cells"]==39
        files[experiment]=hashlib.sha256(qa.read_bytes()).hexdigest()
out={"status":"PASS_LOSS_OPTIMIZER_EVALUATION_THREE_CHECK_QA","qa_sha256":files,
     "allow_capacity_diagnostic":True,"allow_c_wide_or_c_local_training":False,
     "test_metrics_used_for_training_or_selection":False}
path=root/"audits/round51/loss_optimizer_evaluation_complete.json"
path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
PY
