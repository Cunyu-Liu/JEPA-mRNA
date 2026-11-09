#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while ! $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["allow_ripe"]' "$ROOT/audits/round48/route1_decision.json" 2>/dev/null; do sleep 60; done
$PY "$REPO/tools/validate_ripe_cuda_round48.py" --pretrained "$ROOT/weights/structrfm_round48" \
  --out "$ROOT/audits/round48/ripe_cuda_gate.recheck.json" --lengths 515 1024 \
  > "$ROOT/logs/round48/ripe_cuda_gate_recheck.log" 2>&1
cmp "$ROOT/audits/round48/ripe_cuda_gate.json" "$ROOT/audits/round48/ripe_cuda_gate.recheck.json"
$PY "$REPO/tools/prepare_ripe_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/ripe_preregistered.json"
$PY "$REPO/tools/prepare_ripe_round48.py" --root "$ROOT" --out "$ROOT/audits/round48/ripe_preregistered.recheck.json"
cmp "$ROOT/audits/round48/ripe_preregistered.json" "$ROOT/audits/round48/ripe_preregistered.recheck.json"
readarray -t CONFIG < <($PY -c 'import json; d=json.load(open("'"$ROOT"'/audits/round48/ripe_preregistered.json")); print(d["training"]["scorer"]); print(d["training"]["pair_latent_dim"])')
SCORER=${CONFIG[0]}
PAIR_LATENT_DIM=${CONFIG[1]}
SPLITS=(bprna_ts0 bprna_new archiveii ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb rfam_fam chanrg_test chanrg_gena chanrg_genc chanrg_genf)
wait_three() {
  local fail=0 status
  set +e
  for pid in "$@"; do
    wait "$pid"
    status=$?
    if [ "$status" -ne 0 ]; then fail=1; fi
  done
  set -e
  if [ "$fail" -ne 0 ]; then echo "parallel job failed" >&2; return 1; fi
}
train_seed() {
  local seed=$1 out=$ROOT/runs/${ARM}_s${seed}
  mkdir -p "$out"
  if [ -f "$out/run_meta.json" ]; then
    ARM=$ARM MODE=$MODE ENCODER=$ENCODER SEED=$seed $PY -c 'import json,os,pathlib; d=json.loads((pathlib.Path(os.environ["ROOT"])/"runs"/(os.environ["ARM"]+"_s"+os.environ["SEED"])/"run_meta.json").read_text()); assert d["arm"]==os.environ["ARM"]+"_s"+os.environ["SEED"] and d["seed"]==int(os.environ["SEED"]); assert d["ripe"]["mode"]==(None if os.environ["ENCODER"]=="1" else os.environ["MODE"]); assert d["ripe"]["encoder"]==(os.environ["ENCODER"]=="1"); assert d["scorer"]==os.environ["SCORER"] and d["pair_latent_dim"]==int(os.environ["PAIR_LATENT_DIM"])'
    if $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' "$out/run_meta.json" 2>/dev/null; then return 0; fi
  fi
  local extra=()
  if [ "$ENCODER" = 1 ]; then extra=(--ripe-encoder --ripe-encoder-only); fi
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round48" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 --gpu-reserve-gb 3 \
    --scorer "$SCORER" --pair-latent-dim "$PAIR_LATENT_DIM" --ripe-mode "$MODE" "${extra[@]}" \
    >> "$ROOT/logs/round48/${ARM}_s${seed}.log" 2>&1
  CHECKPOINT=$out/best.pt META=$out/run_meta.json $PY -c 'import hashlib,json,os,torch; meta=json.load(open(os.environ["META"])); assert meta["converged"] and meta["early_stopped"] and meta["status"]=="completed"; d=torch.load(os.environ["CHECKPOINT"],map_location="cpu",weights_only=False); assert d["step"]==meta["best_step"] and d["meta"]["scorer"]==meta["scorer"]; print(json.dumps({"arm":meta["arm"],"step":d["step"],"loss":meta["best_validation_loss"],"checkpoint_sha256":hashlib.sha256(open(os.environ["CHECKPOINT"],"rb").read()).hexdigest()}))'
}
eval_seed() {
  local seed=$1 checkpoint=$ROOT/runs/${ARM}_s${seed}/best.pt
  for split in "${SPLITS[@]}"; do
    local out=$ROOT/eval_decision/$ARM/s${seed}/${split}.json
    if [ -f "$out" ] && EXPECT_FULL=$ENCODER $PY -c 'import hashlib,json,os,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["score_forward_count_per_sequence"]==1; assert os.environ["EXPECT_FULL"]!="1" or d["n"]==d["raw_n"]' "$out" "$checkpoint" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 > "$ROOT/logs/round48/eval_${ARM}_s${seed}_${split}.log" 2>&1
  done
  $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
    --pretrained "$ROOT/weights/structrfm_round48" --data "$ROOT/data/validation/vl0_196.jsonl" \
    --out "$ROOT/eval_decision/$ARM/vl0_s${seed}.json" --repeats 3 \
    > "$ROOT/logs/round48/eval_${ARM}_s${seed}_vl0.log" 2>&1
}
ablation_seed() {
  local seed=$1 checkpoint=$ROOT/runs/${ARM}_s${seed}/best.pt
  local experiment=${ARM}_${ABLATION}
  local extra=(--decoder greedy) expected_decoder=greedy expected_prior=none
  if [ "$ABLATION" = turner0 ]; then extra=(--prior-weight-override 0); expected_decoder=nussinov; expected_prior=zero; fi
  for split in "${SPLITS[@]}"; do
    local out=$ROOT/eval_decision/$experiment/s${seed}/${split}.json
    if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert d["checkpoint_sha256"]==h and d["repeats"]==3 and d["device"]=="cuda" and d["decoder"]==sys.argv[3] and d["prior_weight_override"]==(None if sys.argv[4]=="none" else 0.0)' "$out" "$checkpoint" "$expected_decoder" "$expected_prior" 2>/dev/null; then continue; fi
    if [ -f "$out" ]; then mv "$out" "$out.stale"; fi
    $PY "$REPO/tools/eval_structrfm_full_ablation_round48.py" --checkpoint "$checkpoint" \
      --pretrained "$ROOT/weights/structrfm_round48" \
      --data "$ROOT/data/structrfm_pretrain_clean/${split}.jsonl" \
      --out "$out" --repeats 3 "${extra[@]}" \
      > "$ROOT/logs/round48/eval_${experiment}_s${seed}_${split}.log" 2>&1
  done
}
for ARM in ripe_head_log ripe_head_r1 ripe_head_r2 ripe_head_r3 ripe_head_r5 ripe_head_r6 ripe_encoder_r3; do
  MODE=${ARM##*_}
  ENCODER=0
  if [[ "$ARM" == ripe_encoder_* ]]; then ENCODER=1; fi
  export ROOT ARM MODE ENCODER SCORER PAIR_LATENT_DIM
  if $PY - "$ROOT" "$ARM" <<'PY'
import json
import sys
from pathlib import Path
root, arm = Path(sys.argv[1]), sys.argv[2]
for seed in range(3):
    meta = json.loads((root / "runs" / f"{arm}_s{seed}" / "run_meta.json").read_text())
    assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
for suffix in ("", "_greedy", "_turner0"):
    summary = json.loads((root / "eval_decision" / f"{arm}{suffix}" / "summary.json").read_text())
    assert summary["status"] == "PASS_STRUCTRFM_FULL_ABC_13_SPLITS_3_SEEDS_3_REPEATS"
    assert len(summary["splits"]) == 13
    assert all(len(split["cells"]) == 3 for split in summary["splits"].values())
PY
  then
    echo "skip closed $ARM: three converged seeds and three complete summaries"
    continue
  fi
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 21000 ]; do sleep 60; done
  pids=()
  for seed in 0 1 2; do train_seed "$seed" & pids+=("$!"); done
  if ! wait_three "${pids[@]}"; then
    for seed in 0 1 2; do
      if ! $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' "$ROOT/runs/${ARM}_s${seed}/run_meta.json" 2>/dev/null; then
        echo "serial resume for incomplete $ARM seed $seed with unchanged training configuration"
        train_seed "$seed"
      fi
    done
  fi
  pids=()
  for seed in 0 1 2; do eval_seed "$seed" & pids+=("$!"); done
  wait_three "${pids[@]}"
  $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
    --experiment "$ARM" --run-prefix "${ARM}_s" --out "$ROOT/eval_decision/$ARM/summary.json" \
    > "$ROOT/logs/round48/${ARM}_aggregate.log" 2>&1
  $PY "$REPO/tools/analyze_structrfm_confidence_round48.py" --root "$ROOT" --full-model \
    --experiment "$ARM" --run-prefix "${ARM}_s" > "$ROOT/logs/round48/${ARM}_confidence.log" 2>&1
  $PY "$REPO/tools/compare_structrfm_sota_round48.py" --root "$ROOT" --experiment "$ARM" \
    --baseline "$ROOT/source_data/round48/round39_own_clean_leaderboard.json" \
    --rfam-baseline "$ROOT/source_data/round48/round43_bpfold_rfam.json" \
    --chanrg-baseline "$ROOT/source_data/round48/round40_baseline_ood_summary.json" \
    > "$ROOT/logs/round48/${ARM}_sota.log" 2>&1
  for ABLATION in greedy turner0; do
    export ABLATION
    pids=()
    for seed in 0 1 2; do ablation_seed "$seed" & pids+=("$!"); done
    wait_three "${pids[@]}"
    $PY "$REPO/tools/aggregate_structrfm_frozen_round48.py" --root "$ROOT" --full-model \
      --experiment "${ARM}_${ABLATION}" --run-prefix "${ARM}_s" \
      --out "$ROOT/eval_decision/${ARM}_${ABLATION}/summary.json" \
      > "$ROOT/logs/round48/${ARM}_${ABLATION}_aggregate.log" 2>&1
  done
done
