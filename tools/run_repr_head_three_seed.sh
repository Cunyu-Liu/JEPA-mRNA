#!/usr/bin/env bash
# Run one full seed of the frozen three-arm representation/head confirmation.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
SEED="${SEED:-1}"
MODE=full
STEPS=9600
LIMIT=0
STAGE="${STAGE:-$ART/representation_head_mechanism_4arm_v1/seed${SEED}}"
CONTRACT="$REPO/configs/representation_head_mechanism_4arm_v1.json"
INIT="$ART/prior_alignment_v1/init/shared_init_seed${SEED}.pt"
PILOT=/home/rna_jev_artifacts/representation_head_mechanism_4arm_v1/seed0
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
[[ "$SEED" =~ ^[12]$ ]] || { echo "three-seed confirmation accepts only SEED=1 or SEED=2"; exit 2; }
[[ -s "$INIT" ]] || { echo "missing shared init: $INIT"; exit 2; }
cd "$REPO"
"$PY" - "$CONTRACT" "$INIT" "$SEED" "$PILOT" <<'PY' > /tmp/repr_head_preflight_seed.json
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

contract = json.loads(Path(sys.argv[1]).read_text())
init, seed, pilot = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
train = contract["training"]
gate = contract["three_seed_gate"]
pilot_verdict = json.loads((pilot / "evaluation/pilot_verdict.json").read_text())
independent = json.loads((pilot / "final_independent_check.json").read_text())
checks = {
    "data_sha256": sha(train["data"]) == train["data_sha256"],
    "shared_init_sha256": sha(init) == train[f"shared_init_sha256_seed{seed}"],
    "decoder_source_sha256": sha(contract["evaluation"]["decoder_source"]) == contract["evaluation"]["decoder_source_sha256"],
    "evaluation_data_exists": Path(train["evaluation_data"]).exists(),
    "contract_frozen_for_three_seed": contract["status"] == "FROZEN_BEFORE_THREE_SEED",
    "pilot_verdict_sha256": sha(pilot / "evaluation/pilot_verdict.json") == gate["pilot_verdict_sha256"],
    "pilot_direction_confirmed": gate["pilot_direction_confirmed"] and pilot_verdict["decision"] == "PILOT_DIRECTION_CONFIRMED",
    "pilot_independent_pass": independent["overall"] == "PASS",
    "promoted_arms_exact": gate["promoted_arms"] == ["head_deep", "repr_head"],
    "external_closed": contract["three_seed_gate"]["external_evaluation_remains_closed"],
}
result = {
    "checks": checks,
    "all_pass": all(checks.values()),
    "data_sha256": sha(train["data"]),
    "shared_init_sha256": sha(init),
    "decoder_source_sha256": sha(contract["evaluation"]["decoder_source"]),
    "pilot_verdict_sha256": sha(pilot / "evaluation/pilot_verdict.json"),
    "pilot_independent_check_sha256": sha(pilot / "final_independent_check.json"),
}
print(json.dumps(result, indent=2))
assert all(checks.values()), checks
PY
[[ ! -e "$STAGE" ]] || { echo "refusing to overwrite existing stage: $STAGE"; exit 2; }
if ! mkdir -p "$STAGE"; then
  STAGE="/home/rna_jev_artifacts/representation_head_mechanism_4arm_v1/seed${SEED}"
  [[ ! -e "$STAGE" ]] || { echo "refusing to overwrite fallback stage: $STAGE"; exit 2; }
  mkdir -p "$STAGE"
fi
cp /tmp/repr_head_preflight_seed.json "$STAGE/asset_preflight.json"
cp "$CONTRACT" "$STAGE/contract.executed.json"
cp "$0" "$STAGE/command.executed.sh"
git status --short --branch > "$STAGE/git_status_before.txt"
git ls-files --others --exclude-standard > "$STAGE/untracked_before.txt"
git rev-parse HEAD > "$STAGE/git_head_before.txt"
sha256sum tools/train_plan_a_clean_head_p1.py tools/evaluate_repr_head_matrix.py \
  tools/verify_repr_head_matrix.py tools/independent_repr_head_recheck.py \
  src/rnajepa/repr_head_matrix.py "$CONTRACT" > "$STAGE/code_and_contract_sha256.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
RUN_GIT_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA256="$(sha256sum tools/train_plan_a_clean_head_p1.py | cut -d' ' -f1)"
INIT_SHA256="$(sha256sum "$INIT" | cut -d' ' -f1)"
CONTRACT_SHA256="$(sha256sum "$CONTRACT" | cut -d' ' -f1)"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "REPR_HEAD_THREE_SEED_PREFLIGHT_PASS stage=$STAGE seed=$SEED"
  exit 0
fi
gate() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ "$FREE" -ge "$MIN_FREE_MIB" ]] && break
    echo "[gate] free=${FREE}MiB < ${MIN_FREE_MIB}MiB; waiting 60s"
    sleep 60
  done
}
gate
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/gpu_before.txt"
declare -A DZ=( [control_matched]=128 [head_deep]=128 [repr_head]=256 )
declare -A BL=( [control_matched]=4 [head_deep]=8 [repr_head]=8 )
declare -A HD=( [control_matched]=64 [head_deep]=128 [repr_head]=128 )
for ARM in control_matched head_deep repr_head; do
  gate
  echo "=== arm=$ARM d_z=${DZ[$ARM]} blocks=${BL[$ARM]} hidden=${HD[$ARM]} steps=$STEPS ==="
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --teacher-dir "" \
    --contract "$CONTRACT" --contract-mode "$MODE" --train-turner off \
    --raw-score-arm "$ARM" --shared-init "$INIT" \
    --out "$STAGE/$ARM" --snapshot-dir "$STAGE/$ARM/snapshots" \
    --seed "$SEED" --steps "$STEPS" --batch-size 4 --head-lr 1e-4 --backbone-lr 1e-6 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --max-unfrozen-blocks 33 --warmup-steps 5 --grad-clip 1 --nll-normalization length \
    --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
    --lambda-nll 1 --lambda-marginal-bce 0 --marginal-vjp-epsilon 1e-3 \
    --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
    --scorer resnet2d --no-early-stop --save-every 500 --snapshot-every 0 --snapshot-steps "" \
    --d-z "${DZ[$ARM]}" --resnet-blocks "${BL[$ARM]}" --resnet-hidden "${HD[$ARM]}" --head-init fresh \
    --resnet-per-block-checkpoint \
    > "$STAGE/${ARM}.train.stdout.log" 2>&1
  test -s "$STAGE/$ARM/snapshots/${ARM}_step${STEPS}.pt" || {
    echo "training stopped before a usable terminal checkpoint: $ARM"
    tail -25 "$STAGE/${ARM}.train.stdout.log"
    exit 1
  }
  "$PY" tools/export_prior_alignment_scores.py \
    --checkpoint "$STAGE/$ARM/snapshots/${ARM}_step${STEPS}.pt" \
    --train-trajectory train_off --contract-mode "$MODE" \
    --out "$STAGE/$ARM/export_train_off" --limit "$LIMIT" --resume \
    > "$STAGE/${ARM}.export.stdout.log" 2>&1
done
"$PY" tools/evaluate_repr_head_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" \
  --stage "$STAGE" --out "$STAGE/evaluation" --limit "$LIMIT" \
  > "$STAGE/evaluation.stdout.log" 2>&1
RUN_GIT_HEAD="$RUN_GIT_HEAD" TRAIN_SHA256="$TRAIN_SHA256" INIT_SHA256="$INIT_SHA256" \
CONTRACT_SHA256="$CONTRACT_SHA256" STAGE="$STAGE" MODE="$MODE" STEPS="$STEPS" SEED="$SEED" \
"$PY" - <<'PY2'
import json
import os
from pathlib import Path
stage = Path(os.environ["STAGE"])
verdict = json.loads((stage / "evaluation/pilot_verdict.json").read_text())
provenance = {key: os.environ[key] for key in (
    "RUN_GIT_HEAD", "TRAIN_SHA256", "INIT_SHA256", "CONTRACT_SHA256", "MODE", "STEPS", "SEED")}
provenance.update({
    "decision": verdict["decision"],
    "promoted_arms": verdict["promoted_arms"],
    "external_evaluation_allowed": False,
    "control_arm": "control_matched",
    "full_control_reused_not_retrained": False,
    "artifact_staging": str(stage),
})
(stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
PY2
"$PY" tools/verify_repr_head_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" --stage "$STAGE" \
  --evaluation "$STAGE/evaluation" --out "$STAGE/verification.json" \
  > "$STAGE/verification.stdout.log" 2>&1
"$PY" tools/independent_repr_head_recheck.py \
  --contract "$CONTRACT" --stage "$STAGE" --evaluation "$STAGE/evaluation" \
  --out "$STAGE/final_independent_check.json" --seed "$SEED" --steps "$STEPS" \
  > "$STAGE/recheck.stdout.log" 2>&1
echo "REPR_HEAD_THREE_SEED_ARM_COMPLETE stage=$STAGE"
