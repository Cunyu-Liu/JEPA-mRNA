#!/usr/bin/env bash
# Resume only the failed marginal arm; frozen and limited4 remain immutable.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
SEED=0
STEPS=9600
STAGE="$ART/raw_score_mechanism_4arm_v1/seed0"
CONTRACT="$REPO/configs/raw_score_mechanism_4arm_v1.json"
INIT="$ART/prior_alignment_v1/init/shared_init_seed0.pt"
cd "$REPO"
for ARM in frozen limited4; do
  "$PY" - "$STAGE/$ARM/run_meta.json" "$ARM" <<'PY2'
import json, sys
meta = json.load(open(sys.argv[1]))
assert meta["status"] == "completed", sys.argv[2]
assert meta["steps_completed"] == 9600 and meta["stop_reason"] == "step_cap", sys.argv[2]
PY2
done
[[ -s "$STAGE/marginal/resume.pt" ]] || { echo "missing marginal resume checkpoint"; exit 2; }
[[ ! -e "$STAGE/evaluation" ]] || { echo "evaluation already exists"; exit 2; }
[[ ! -e "$STAGE/marginal/export_train_off" ]] || { echo "marginal export already exists"; exit 2; }
RESUME_STEP="$("$PY" - "$STAGE/marginal/resume.pt" <<'PY2'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
step = int(state["step"])
assert 0 < step < 9600, step
print(step)
PY2
)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
RUN_GIT_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA256="$(sha256sum tools/train_plan_a_clean_head_p1.py | cut -d' ' -f1)"
MARGINAL_SHA256="$(sha256sum src/rnajepa/structured_marginal.py | cut -d' ' -f1)"
INIT_SHA256="$(sha256sum "$INIT" | cut -d' ' -f1)"
CONTRACT_SHA256="$(sha256sum "$CONTRACT" | cut -d' ' -f1)"
printf '\n[engineering-resume] from_step=%s git_head=%s empty_legal_set_patch=2b339dab37fc7b7f0e27127a3908ad808f6cd77c\n' \
  "$RESUME_STEP" "$RUN_GIT_HEAD" >> "$STAGE/marginal.train.stdout.log"
"$PY" tools/train_plan_a_clean_head_p1.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir "" \
  --contract "$CONTRACT" --contract-mode pilot --train-turner off \
  --raw-score-arm marginal --shared-init "$INIT" \
  --out "$STAGE/marginal" --snapshot-dir "$STAGE/marginal/snapshots" \
  --seed "$SEED" --steps "$STEPS" --batch-size 4 \
  --head-lr 1e-4 --backbone-lr 1e-6 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --max-unfrozen-blocks 33 \
  --warmup-steps 5 --grad-clip 1 --nll-normalization length \
  --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
  --lambda-nll 1 --lambda-marginal-bce 1 --marginal-vjp-epsilon 1e-3 \
  --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
  --scorer resnet2d --no-early-stop --save-every 500 \
  --snapshot-every 0 --snapshot-steps "" \
  >> "$STAGE/marginal.train.stdout.log" 2>&1
test -s "$STAGE/marginal/snapshots/marginal_step9600.pt"
"$PY" tools/export_prior_alignment_scores.py \
  --checkpoint "$STAGE/marginal/snapshots/marginal_step9600.pt" \
  --train-trajectory train_off --contract-mode pilot \
  --out "$STAGE/marginal/export_train_off" --limit 0 --resume \
  > "$STAGE/marginal.export.stdout.log" 2>&1
"$PY" tools/evaluate_raw_score_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" \
  --stage "$STAGE" --out "$STAGE/evaluation" --limit 0 \
  > "$STAGE/evaluation.stdout.log" 2>&1
RUN_GIT_HEAD="$RUN_GIT_HEAD" TRAIN_SHA256="$TRAIN_SHA256" \
MARGINAL_SHA256="$MARGINAL_SHA256" INIT_SHA256="$INIT_SHA256" \
CONTRACT_SHA256="$CONTRACT_SHA256" STAGE="$STAGE" RESUME_STEP="$RESUME_STEP" \
"$PY" - <<'PY2'
import json, os
from pathlib import Path
stage = Path(os.environ["STAGE"])
verdict = json.loads((stage / "evaluation/pilot_verdict.json").read_text())
provenance = {k: os.environ[k] for k in (
    "RUN_GIT_HEAD", "TRAIN_SHA256", "MARGINAL_SHA256", "INIT_SHA256",
    "CONTRACT_SHA256", "RESUME_STEP")}
provenance.update({
    "MODE": "pilot", "STEPS": "9600", "SEED": "0",
    "INITIAL_MATRIX_GIT_HEAD": "84913d4af958d270aa0b71f88c17cc9f879ba8b9",
    "MARGINAL_EMPTY_LEGAL_SET_PATCH": "2b339dab37fc7b7f0e27127a3908ad808f6cd77c",
    "engineering_note": "Only the marginal arm resumed after defining an empty legal-pair set as differentiable zero marginal supervision; NLL and all nonempty-set calculations are unchanged.",
    "decision": verdict["decision"], "promoted_arms": verdict["promoted_arms"],
    "external_evaluation_allowed": False, "full_control_reused_not_retrained": True,
})
(stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
PY2
