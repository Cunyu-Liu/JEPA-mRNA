#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
CONFIG=$ROOT/data/rna_complex_round49/d1_preparation/d1_config.json
while [[ ! -f "$CONFIG" ]]; do sleep 60; done
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS_COMPLEX_D1_PRECONFIGURED"' "$CONFIG"
readarray -t C < <($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["d0"]["run_prefix"]); print(d["d1"]["data"]); print(d["d1"]["teacher_dir"]); print(d["complex_holdout"]); print(d["d1"]["scorer"]); print(d["d1"]["pair_latent_dim"]); print(d["d1"]["ripe"]["mode"] or ""); print(int(d["d1"]["ripe"]["encoder"])); print(d["d1"]["objective"]["lambda_jepa"]); print(d["d1"]["objective"]["jepa_target"] or "none")' "$CONFIG")
D0_PREFIX=${C[0]}
DATA=${C[1]}
TEACHER=${C[2]}
HOLDOUT=${C[3]}
SCORER=${C[4]}
PAIR_LATENT_DIM=${C[5]}
RIPE_MODE=${C[6]}
RIPE_ENCODER=${C[7]}
LAMBDA_JEPA=${C[8]}
TARGET=${C[9]}
export ROOT DATA TEACHER HOLDOUT SCORER PAIR_LATENT_DIM RIPE_MODE RIPE_ENCODER LAMBDA_JEPA TARGET D0_PREFIX
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
wait_jobs() { local fail=0 status; set +e; for pid in "$@"; do wait "$pid"; status=$?; if [ "$status" -ne 0 ]; then fail=1; fi; done; set -e; if [ "$fail" -ne 0 ]; then return 1; fi; }
train_seed() {
  local seed=$1 out=$ROOT/runs/complex_d1_s${seed}
  mkdir -p "$out"
  if [ -f "$out/run_meta.json" ] && $PY -c 'import json,os,sys; d=json.load(open(sys.argv[1])); assert d["data"]==os.environ["DATA"]; assert d["seed"]==int(sys.argv[2]) and d["scorer"]==os.environ["SCORER"] and d["pair_latent_dim"]==int(os.environ["PAIR_LATENT_DIM"]); assert d["objective"]["lambda_jepa"]==float(os.environ["LAMBDA_JEPA"])' "$out/run_meta.json" "$seed" 2>/dev/null; then
    if $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' "$out/run_meta.json" 2>/dev/null; then return 0; fi
  fi
  local extra=(--lambda-jepa "$LAMBDA_JEPA" --jepa-target "$TARGET")
  if [ "$SCORER" = pairmlp ]; then extra=(); fi
  if [ "$RIPE_ENCODER" = 1 ]; then extra+=(--ripe-encoder --ripe-encoder-only); fi
  local ripe_args=()
  if [ -n "$RIPE_MODE" ]; then ripe_args=(--ripe-mode "$RIPE_MODE"); fi
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data "$DATA" --teacher-dir "$TEACHER" --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 --save-every 500 \
    --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round48" --warmup-head-steps 1600 \
    --unfreeze-every 800 --unfreeze-per-step 2 --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 \
    --gpu-reserve-gb 3 --scorer "$SCORER" --pair-latent-dim "$PAIR_LATENT_DIM" "${ripe_args[@]}" "${extra[@]}" \
    >> "$ROOT/logs/round48/complex_d1_s${seed}.log" 2>&1
}
eval_checkpoint() {
  local checkpoint=$1 experiment=$2 seed=$3 split=$4 input=$5 out=$ROOT/eval_decision/$experiment/s${seed}/${split}.json
  if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint_sha256"]==hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest() and d["data_sha256"]==hashlib.sha256(open(sys.argv[3],"rb").read()).hexdigest() and d["repeats"]==3 and d["device"]=="cuda"' "$out" "$checkpoint" "$input" 2>/dev/null; then return 0; fi
  if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
  $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
    --pretrained "$ROOT/weights/structrfm_round48" --data "$input" --out "$out" --repeats 3 \
    > "$ROOT/logs/round48/eval_${experiment}_s${seed}_${split}.log" 2>&1
}
pids=()
for seed in 0 1 2; do
  eval_checkpoint "$ROOT/runs/${D0_PREFIX}${seed}/best.pt" complex_d0 "$seed" complex_holdout "$HOLDOUT" & pids+=("$!")
done
wait_jobs "${pids[@]}"
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 16000 ]; do sleep 60; done
pids=()
for seed in 0 1; do train_seed "$seed" & pids+=("$!"); done
wait_jobs "${pids[@]}"
train_seed 2
pids=()
for seed in 0 1 2; do
  for split in "${SPLITS[@]}"; do eval_checkpoint "$ROOT/runs/complex_d1_s${seed}/best.pt" complex_d1 "$seed" "$split" "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl"; done &
  eval_checkpoint "$ROOT/runs/complex_d1_s${seed}/best.pt" complex_d1 "$seed" complex_holdout "$HOLDOUT"
  pids+=("$!")
done
wait_jobs "${pids[@]}"
$PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
  --experiment complex_d1 --run-prefix complex_d1_s --out "$ROOT/eval_decision/complex_d1/summary.json" > "$ROOT/logs/round48/complex_d1_aggregate.log" 2>&1
$PY "$REPO/tools/analyze_structrfm_confidence_round48.py" --root "$ROOT" --full-model \
  --experiment complex_d1 --run-prefix complex_d1_s > "$ROOT/logs/round48/complex_d1_confidence.log" 2>&1
$PY "$REPO/tools/compare_structrfm_sota_round48.py" --root "$ROOT" --experiment complex_d1 \
  --baseline "$ROOT/source_data/round48/round39_own_clean_leaderboard.json" \
  --rfam-baseline "$ROOT/source_data/round48/round43_bpfold_rfam.json" \
  --chanrg-baseline "$ROOT/source_data/round48/round40_baseline_ood_summary.json" > "$ROOT/logs/round48/complex_d1_sota.log" 2>&1
for ABLATION in greedy turner0; do
  export ABLATION
  pids=()
  for seed in 0 1 2; do
    for split in "${SPLITS[@]}"; do
      extra=(--decoder greedy)
      if [ "$ABLATION" = turner0 ]; then extra=(--prior-weight-override 0); fi
      $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$ROOT/runs/complex_d1_s${seed}/best.pt" \
        --pretrained "$ROOT/weights/structrfm_round48" --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
        --out "$ROOT/eval_decision/complex_d1_${ABLATION}/s${seed}/${split}.json" --repeats 3 "${extra[@]}" \
        > "$ROOT/logs/round48/eval_complex_d1_${ABLATION}_s${seed}_${split}.log" 2>&1
    done & pids+=("$!")
  done
  wait_jobs "${pids[@]}"
  $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
    --experiment "complex_d1_${ABLATION}" --run-prefix complex_d1_s \
    --out "$ROOT/eval_decision/complex_d1_${ABLATION}/summary.json" > "$ROOT/logs/round48/complex_d1_${ABLATION}_aggregate.log" 2>&1
done
$PY "$REPO/tools/compare_ripe_paired_round48.py" --root "$ROOT" \
  --manifest data/rna_complex_round49/d1_preparation/comparison_manifest.json \
  --status PASS_COMPLEX_D1_PAIRED_MICRO_F1_BOOTSTRAP \
  --out "$ROOT/eval_decision/complex_d1_paired_bootstrap.json" > "$ROOT/logs/round48/complex_d1_paired.log" 2>&1
$PY "$REPO/tools/adjudicate_complex_d1_round49.py" --root "$ROOT" \
  --out "$ROOT/audits/round48/complex_d1_decision.json" > "$ROOT/logs/round48/complex_d1_decision.log" 2>&1
