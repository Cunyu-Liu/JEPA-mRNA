#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
SEED="${SEED:-0}"
MODE="${MODE:-pilot}"
STEPS="${STEPS:-9600}"
LIMIT="${LIMIT:-0}"
STAGE="${STAGE:-$ART/raw_score_mechanism_4arm_v1/seed${SEED}}"
CONTRACT="$REPO/configs/raw_score_mechanism_4arm_v1.json"
INIT="$ART/prior_alignment_v1/init/shared_init_seed${SEED}.pt"
[[ -s "$INIT" ]] || { echo "missing shared init: $INIT"; exit 2; }
cd "$REPO"
ASSET_REPORT="$("$PY" tools/validate_raw_score_assets.py --contract "$CONTRACT" --seed "$SEED")"
[[ ! -e "$STAGE" ]] || { echo "refusing to overwrite existing stage: $STAGE"; exit 2; }
mkdir -p "$STAGE"
printf '%s\n' "$ASSET_REPORT" > "$STAGE/asset_preflight.json"
cp "$CONTRACT" "$STAGE/contract.executed.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
RUN_GIT_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA256="$(sha256sum tools/train_plan_a_clean_head_p1.py | cut -d' ' -f1)"
INIT_SHA256="$(sha256sum "$INIT" | cut -d' ' -f1)"
CONTRACT_SHA256="$(sha256sum "$CONTRACT" | cut -d' ' -f1)"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/gpu_before.txt"

for ARM in frozen limited4 marginal; do
  case "$ARM" in
    frozen) CAP=0; LAMBDA=0 ;;
    limited4) CAP=4; LAMBDA=0 ;;
    marginal) CAP=33; LAMBDA=1 ;;
  esac
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir "" \
    --contract "$CONTRACT" --contract-mode "$MODE" --train-turner off \
    --raw-score-arm "$ARM" --shared-init "$INIT" \
    --out "$STAGE/$ARM" --snapshot-dir "$STAGE/$ARM/snapshots" \
    --seed "$SEED" --steps "$STEPS" --batch-size 4 \
    --head-lr 1e-4 --backbone-lr 1e-6 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --max-unfrozen-blocks "$CAP" \
    --warmup-steps 5 --grad-clip 1 --nll-normalization length \
    --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
    --lambda-nll 1 --lambda-marginal-bce "$LAMBDA" --marginal-vjp-epsilon 1e-3 \
    --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
    --scorer resnet2d --no-early-stop --save-every 500 \
    --snapshot-every 0 --snapshot-steps "" \
    > "$STAGE/${ARM}.train.stdout.log" 2>&1
  test -s "$STAGE/$ARM/snapshots/${ARM}_step${STEPS}.pt" || {
    echo "training stopped before a usable terminal checkpoint: $ARM"; exit 1; }
  "$PY" tools/export_prior_alignment_scores.py \
    --checkpoint "$STAGE/$ARM/snapshots/${ARM}_step${STEPS}.pt" \
    --train-trajectory train_off --contract-mode "$MODE" \
    --out "$STAGE/$ARM/export_train_off" --limit "$LIMIT" --resume \
    > "$STAGE/${ARM}.export.stdout.log" 2>&1
done

"$PY" tools/evaluate_raw_score_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" \
  --stage "$STAGE" --out "$STAGE/evaluation" --limit "$LIMIT" \
  > "$STAGE/evaluation.stdout.log" 2>&1

RUN_GIT_HEAD="$RUN_GIT_HEAD" TRAIN_SHA256="$TRAIN_SHA256" INIT_SHA256="$INIT_SHA256" \
CONTRACT_SHA256="$CONTRACT_SHA256" STAGE="$STAGE" MODE="$MODE" STEPS="$STEPS" SEED="$SEED" \
"$PY" - <<'PY2'
import json, os
from pathlib import Path
stage=Path(os.environ['STAGE'])
verdict=json.loads((stage/'evaluation/pilot_verdict.json').read_text())
provenance={k:os.environ[k] for k in ('RUN_GIT_HEAD','TRAIN_SHA256','INIT_SHA256','CONTRACT_SHA256','MODE','STEPS','SEED')}
provenance.update({'decision':verdict['decision'],'promoted_arms':verdict['promoted_arms'],
                   'external_evaluation_allowed':False,'full_control_reused_not_retrained':True})
(stage/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
PY2
