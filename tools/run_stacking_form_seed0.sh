#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/stacking_closure_form_v1
CONTRACT="$REPO/configs/stacking_closure_form_v1.json"
MODE="${MODE:-smoke}"
if [[ "$MODE" == smoke ]]; then STEPS=2; LIMIT=2; elif [[ "$MODE" == pilot ]]; then STEPS=9600; LIMIT=0; else echo "only smoke/pilot" >&2; exit 2; fi
STAGE="${STAGE:-$ART/$MODE}"
[[ ! -e "$STAGE" ]] || { echo "refusing to overwrite stage: $STAGE" >&2; exit 2; }
mkdir -p "$STAGE"
cd "$REPO"
git rev-parse HEAD > "$STAGE/git_head_before.txt"
git status --short --branch > "$STAGE/git_status_before.txt"
cp "$CONTRACT" "$STAGE/contract.executed.json"
cp "$0" "$STAGE/command.executed.sh"
sha256sum src/rnajepa/stacking_closure.py src/rnajepa/stacking_form_matrix.py src/rnajepa/decision_head.py tools/train_plan_a_clean_head_p1.py "$CONTRACT" > "$STAGE/code_and_contract_sha256.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
for ARM in local_control stacking_closure destroy_offset destroy_channel_shuffle; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [[ "$FREE" -ge 20000 ]] || { echo "GPU free $FREE MiB < 20000" >&2; exit 2; }
  OUT="$STAGE/$ARM"
  [[ ! -e "$OUT" ]] || { echo "refusing to overwrite arm: $OUT" >&2; exit 2; }
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --teacher-dir "" \
    --contract "$CONTRACT" --contract-mode "$MODE" --train-turner off \
    --raw-score-arm "$ARM" --shared-init /teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/init/shared_init_seed0.pt \
    --out "$OUT" --snapshot-dir "$OUT/snapshots" --seed 0 --steps "$STEPS" --batch-size 4 \
    --zchunk 16 --head-lr 1e-4 --backbone-lr 1e-6 --warmup-head-steps 1600 \
    --unfreeze-every 800 --unfreeze-per-step 2 --max-unfrozen-blocks 33 --warmup-steps 5 \
    --grad-clip 1 --nll-normalization length --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
    --lambda-nll 1 --lambda-marginal-bce 0 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 \
    --lambda-jepa 0 --scorer "$ARM" --d-z 128 --resnet-blocks 1 --resnet-hidden 128 \
    --head-init fresh --no-early-stop --save-every 500 --snapshot-every 0 --snapshot-steps "" \
    > "$STAGE/${ARM}.train.stdout.log" 2>&1
  "$PY" tools/export_prior_alignment_scores.py \
    --checkpoint "$OUT/snapshots/${ARM}_step${STEPS}.pt" --train-trajectory train_off \
    --contract-mode "$MODE" --out "$OUT/export_train_off" --limit "$LIMIT" --resume \
    > "$STAGE/${ARM}.export.stdout.log" 2>&1
  echo "STACKING_FORM_ARM_COMPLETE mode=$MODE arm=$ARM"
done
