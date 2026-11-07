#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while ! $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["allow_pair_jepa"]' "$ROOT/audits/round48/ripe_decision.json" 2>/dev/null; do sleep 60; done
$PY "$REPO/tools/prepare_pairjepa_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/pairjepa_preregistered.json" > "$ROOT/logs/round48/pairjepa_prepare.log" 2>&1
$PY "$REPO/tools/prepare_pairjepa_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/pairjepa_preregistered.recheck.json" >> "$ROOT/logs/round48/pairjepa_prepare.log" 2>&1
cmp "$ROOT/audits/round48/pairjepa_preregistered.json" "$ROOT/audits/round48/pairjepa_preregistered.recheck.json"
readarray -t CONFIG < <($PY -c 'import json; d=json.load(open("'"$ROOT"'/audits/round48/pairjepa_preregistered.json")); print(d["ripe_mode"] or ""); print(int(d["ripe_encoder"])); print(d["training"]["pair_latent_dim"])')
RIPE_MODE=${CONFIG[0]}
RIPE_ENCODER=${CONFIG[1]}
PAIR_LATENT_DIM=${CONFIG[2]}
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
wait_jobs() {
  local fail=0 status
  set +e
  for pid in "$@"; do wait "$pid"; status=$?; if [ "$status" -ne 0 ]; then fail=1; fi; done
  set -e
  if [ "$fail" -ne 0 ]; then echo "parallel job failed" >&2; return 1; fi
}
train_seed() {
  local seed=$1 out=$ROOT/runs/${ARM}_s${seed}
  mkdir -p "$out"
  if [ -f "$out/run_meta.json" ]; then
    SEED=$seed $PY -c 'import json,os,pathlib; d=json.loads((pathlib.Path(os.environ["ROOT"])/"runs"/(os.environ["ARM"]+"_s"+os.environ["SEED"])/"run_meta.json").read_text()); assert d["arm"]==os.environ["ARM"]+"_s"+os.environ["SEED"] and d["seed"]==int(os.environ["SEED"]); assert d["scorer"]==os.environ["SCORER"] and d["pair_latent_dim"]==int(os.environ["PAIR_LATENT_DIM"]); assert d["objective"]["jepa_target"]==(None if os.environ["TARGET"]=="none" else os.environ["TARGET"]); assert d["ripe"]["mode"]==(None if os.environ["RIPE_ENCODER"]=="1" else (os.environ["RIPE_MODE"] or None)) and d["ripe"]["encoder"]==(os.environ["RIPE_ENCODER"]=="1")'
    if $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' "$out/run_meta.json" 2>/dev/null; then return 0; fi
  fi
  local extra=(--lambda-jepa 0.1 --jepa-target "$TARGET")
  if [ "$SCORER" = pairmlp ]; then extra=(); fi
  if [ "$RIPE_ENCODER" = 1 ]; then extra+=(--ripe-encoder --ripe-encoder-only); fi
  local ripe_args=()
  if [ -n "$RIPE_MODE" ]; then ripe_args=(--ripe-mode "$RIPE_MODE"); fi
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round48" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 --gpu-reserve-gb 3 \
    --scorer "$SCORER" --pair-latent-dim "$PAIR_LATENT_DIM" "${ripe_args[@]}" "${extra[@]}" \
    >> "$ROOT/logs/round48/${ARM}_s${seed}.log" 2>&1
}
eval_seed() {
  local seed=$1 checkpoint=$ROOT/runs/${ARM}_s${seed}/best.pt
  for split in "${SPLITS[@]}"; do
    local out=$ROOT/eval_decision/$ARM/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,os,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["score_forward_count_per_sequence"]==1; assert d["pairjepa_teacher_forward_guard"]==(os.environ["SCORER"]=="pairjepa"); assert os.environ["RIPE_ENCODER"]!="1" or d["n"]==d["raw_n"]' "$out" "$checkpoint" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 > "$ROOT/logs/round48/eval_${ARM}_s${seed}_${split}.log" 2>&1
  done
  $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
    --pretrained "$ROOT/weights/structrfm_round48" --data "$ROOT/data/validation/vl0_196.jsonl" \
    --out "$ROOT/eval_decision/$ARM/vl0_s${seed}.json" --repeats 3 > "$ROOT/logs/round48/eval_${ARM}_s${seed}_vl0.log" 2>&1
}
ablation_seed() {
  local seed=$1 checkpoint=$ROOT/runs/${ARM}_s${seed}/best.pt experiment=${ARM}_${ABLATION}
  local extra=(--decoder greedy) expected_decoder=greedy expected_prior=none
  if [ "$ABLATION" = turner0 ]; then extra=(--prior-weight-override 0); expected_decoder=nussinov; expected_prior=zero; fi
  for split in "${SPLITS[@]}"; do
    local out=$ROOT/eval_decision/$experiment/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["decoder"]==sys.argv[3] and d["prior_weight_override"]==(None if sys.argv[4]=="none" else 0.0)' "$out" "$checkpoint" "$expected_decoder" "$expected_prior" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 "${extra[@]}" > "$ROOT/logs/round48/eval_${experiment}_s${seed}_${split}.log" 2>&1
  done
}
readarray -t ARMS < <($PY -c 'import json; d=json.load(open("'"$ROOT"'/audits/round48/pairjepa_preregistered.json")); [print(a["arm"],a["scorer"],a["target"] or "none",int(a["reuse_baseline"])) for a in d["arms"]]')
for row in "${ARMS[@]}"; do
  read -r ARM SCORER TARGET REUSE <<< "$row"
  if [ "$REUSE" = 1 ]; then continue; fi
  export ROOT ARM SCORER TARGET RIPE_MODE RIPE_ENCODER PAIR_LATENT_DIM
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 16000 ]; do sleep 60; done
  pids=()
  for seed in 0 1; do train_seed "$seed" & pids+=("$!"); done
  wait_jobs "${pids[@]}"
  train_seed 2
  pids=()
  for seed in 0 1 2; do eval_seed "$seed" & pids+=("$!"); done
  wait_jobs "${pids[@]}"
  $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
    --experiment "$ARM" --run-prefix "${ARM}_s" --out "$ROOT/eval_decision/$ARM/summary.json" > "$ROOT/logs/round48/${ARM}_aggregate.log" 2>&1
  $PY "$REPO/tools/analyze_structrfm_confidence_round48.py" --root "$ROOT" --full-model \
    --experiment "$ARM" --run-prefix "${ARM}_s" > "$ROOT/logs/round48/${ARM}_confidence.log" 2>&1
  $PY "$REPO/tools/compare_structrfm_sota_round48.py" --root "$ROOT" --experiment "$ARM" \
    --baseline "$ROOT/source_data/round48/round39_own_clean_leaderboard.json" \
    --rfam-baseline "$ROOT/source_data/round48/round43_bpfold_rfam.json" \
    --chanrg-baseline "$ROOT/source_data/round48/round40_baseline_ood_summary.json" > "$ROOT/logs/round48/${ARM}_sota.log" 2>&1
  for ABLATION in greedy turner0; do
    export ABLATION
    pids=()
    for seed in 0 1 2; do ablation_seed "$seed" & pids+=("$!"); done
    wait_jobs "${pids[@]}"
    $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
      --experiment "${ARM}_${ABLATION}" --run-prefix "${ARM}_s" \
      --out "$ROOT/eval_decision/${ARM}_${ABLATION}/summary.json" > "$ROOT/logs/round48/${ARM}_${ABLATION}_aggregate.log" 2>&1
  done
done
