#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_prior_alignment_stage.sh smoke|pilot|full SEED}"
SEED="${2:?usage: run_prior_alignment_stage.sh smoke|pilot|full SEED}"
REPO="${REPO:-/home/rna_jev_xens2_dualview_round56}"
PY="${PY:-/root/miniconda3/envs/lucaone_cu128/bin/python}"
ART="${ART:-/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1}"

case "$MODE" in
  smoke) STEPS=2; LIMIT=2; SAVE_EVERY=0 ;;
  pilot) STEPS=3200; LIMIT=0; SAVE_EVERY=500; [[ "$SEED" == "0" ]] || { echo "pilot is frozen to seed0"; exit 2; } ;;
  full) STEPS=20000; LIMIT=0; SAVE_EVERY=500; [[ "$SEED" =~ ^[012]$ ]] || { echo "full seeds are 0/1/2"; exit 2; } ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac

STAGE="$ART/${MODE}_seed${SEED}"
INIT="$ART/init/shared_init_seed${SEED}.pt"
mkdir -p "$STAGE" "$ART/init" "$STAGE/snapshots"
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader

if [[ ! -s "$INIT" ]]; then
  "$PY" tools/build_prior_alignment_shared_init.py --seed "$SEED" --out "$INIT" \
    > "$STAGE/shared_init.stdout.log" 2>&1
fi

for ARM in on off; do
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir "" \
    --contract-mode "$MODE" --train-turner "$ARM" --shared-init "$INIT" \
    --out "$STAGE/train_$ARM" --snapshot-dir "$STAGE/snapshots" \
    --seed "$SEED" --steps "$STEPS" --batch-size 4 \
    --head-lr 1e-4 --backbone-lr 1e-5 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --warmup-steps 5 --grad-clip 1 --nll-normalization length \
    --lambda-nll 1 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
    --scorer resnet2d --no-early-stop --save-every "$SAVE_EVERY" \
    > "$STAGE/train_${ARM}.stdout.log" 2>&1
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
  --contract-mode "$MODE" --out "$STAGE/decode_vl0" \
  > "$STAGE/decode_vl0.stdout.log" 2>&1

"$PY" tools/verify_prior_alignment_stage.py \
  --stage "$STAGE" --mode "$MODE" --seed "$SEED" \
  --out "$STAGE/stage_verification.json" \
  > "$STAGE/stage_verification.stdout.log" 2>&1

STAGE="$STAGE" MODE="$MODE" SEED="$SEED" REPO="$REPO" "$PY" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

stage = Path(os.environ["STAGE"])
repo = Path(os.environ["REPO"])
contract = repo / "configs/prior_alignment_2x2_v1.json"
manifest = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "mode": os.environ["MODE"],
    "seed": int(os.environ["SEED"]),
    "status": "COMPLETED",
    "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "old_external_panels_used": False,
    "artifacts": {
        "train_on": str(stage / "train_on"),
        "train_off": str(stage / "train_off"),
        "decode_vl0": str(stage / "decode_vl0"),
        "verification": str(stage / "stage_verification.json"),
        "decode_verification": str(stage / "decode_vl0/decode_verification.json"),
    },
}
(stage / "stage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
