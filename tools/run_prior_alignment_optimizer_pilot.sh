#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: run_prior_alignment_optimizer_pilot.sh SEED}"
[[ "$SEED" =~ ^[012]$ ]] || { echo "optimizer pilot seeds are 0/1/2"; exit 2; }
REPO="${REPO:-/home/rna_jev_xens2_dualview_round56}"
PY="${PY:-/root/miniconda3/envs/lucaone_cu128/bin/python}"
BASE="${BASE:-/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1}"
ART="${ART:-/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_optimizer_v1}"
STAGE="$ART/seed${SEED}_backbone_lr_1e-6"
INIT="$BASE/init/shared_init_seed${SEED}.pt"
MODE="${MODE:-pilot}"
STEPS="${STEPS:-9600}"
SNAPSHOT_STEPS="${SNAPSHOT_STEPS-3200,6400}"
LIMIT="${LIMIT:-0}"

[[ -s "$INIT" ]] || { echo "missing shared init: $INIT"; exit 2; }
[[ ! -e "$STAGE" ]] || { echo "refusing to overwrite existing stage: $STAGE"; exit 2; }
mkdir -p "$STAGE/snapshots"
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_GIT_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA256="$(sha256sum tools/train_plan_a_clean_head_p1.py | awk '{print $1}')"
INIT_SHA256="$(sha256sum "$INIT" | awk '{print $1}')"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader

for ARM in on off; do
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir "" \
    --contract "$REPO/configs/prior_alignment_optimizer_v1.json" \
    --contract-mode "$MODE" --train-turner "$ARM" --shared-init "$INIT" \
    --out "$STAGE/train_$ARM" --snapshot-dir "$STAGE/snapshots" \
    --seed "$SEED" --steps "$STEPS" --batch-size 4 \
    --head-lr 1e-4 --backbone-lr 1e-6 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --warmup-steps 5 --grad-clip 1 --nll-normalization length \
    --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
    --lambda-nll 1 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
    --scorer resnet2d --no-early-stop --save-every 500 \
    --snapshot-every 0 --snapshot-steps "$SNAPSHOT_STEPS" \
    > "$STAGE/train_${ARM}.stdout.log" 2>&1
  test -s "$STAGE/snapshots/train_${ARM}_step${STEPS}.pt" || {
    echo "training stopped before a usable terminal checkpoint: $ARM"
    exit 1
  }
done

for ARM in on off; do
  "$PY" tools/export_prior_alignment_scores.py \
    --checkpoint "$STAGE/snapshots/train_${ARM}_step${STEPS}.pt" \
    --train-trajectory "train_$ARM" --contract-mode "$MODE" \
    --out "$STAGE/export_train_$ARM" --limit "$LIMIT" --resume \
    > "$STAGE/export_train_${ARM}.stdout.log" 2>&1
done

"$PY" tools/decode_prior_alignment_vl0.py \
  --train-on-export "$STAGE/export_train_on" \
  --train-off-export "$STAGE/export_train_off" \
  --contract "$REPO/configs/prior_alignment_optimizer_v1.json" \
  --contract-mode "$MODE" --out "$STAGE/decode_vl0" \
  > "$STAGE/decode_vl0.stdout.log" 2>&1

STAGE="$STAGE" SEED="$SEED" STEPS="$STEPS" RUN_GIT_HEAD="$RUN_GIT_HEAD" TRAIN_SHA256="$TRAIN_SHA256" INIT_SHA256="$INIT_SHA256" "$PY" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

stage = Path(os.environ["STAGE"])
summary = json.loads((stage / "decode_vl0/vl0_selected_summary.json").read_text())
manifest = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "COMPLETED",
    "scope": "SINGLE_FACTOR_BACKBONE_LR_STABILITY_PILOT_ONLY",
    "seed": int(os.environ["SEED"]),
    "steps": int(os.environ["STEPS"]),
    "backbone_lr": 1e-6,
    "head_lr": 1e-4,
    "baseline_backbone_lr": 1e-5,
    "git_head": os.environ["RUN_GIT_HEAD"],
    "train_sha256": os.environ["TRAIN_SHA256"],
    "shared_init_sha256": os.environ["INIT_SHA256"],
    "old_external_panels_used": False,
    "production_repair": False,
    "sota_claim": False,
    "arms": summary["arms"],
}
(stage / "optimizer_pilot_verdict.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
