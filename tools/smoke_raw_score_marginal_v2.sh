#!/usr/bin/env bash
# Engineering-only v2 smoke: train only marginal; inherit frozen/limited4 smoke runs.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
V1_SMOKE="$ART/raw_score_mechanism_smoke_v1/seed0"
STAGE="$ART/raw_score_mechanism_v2_smoke/seed0"
CONTRACT="$REPO/configs/raw_score_mechanism_4arm_v2.json"
INIT="$ART/prior_alignment_v1/init/shared_init_seed0.pt"
cd "$REPO"
git diff --quiet
git diff --cached --quiet
[[ ! -e "$STAGE" ]] || { echo "refusing to overwrite v2 smoke: $STAGE"; exit 2; }
mkdir -p "$STAGE"
ln -s "$V1_SMOKE/frozen" "$STAGE/frozen"
ln -s "$V1_SMOKE/limited4" "$STAGE/limited4"
cp "$CONTRACT" "$STAGE/contract.executed.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/gpu_before.txt"
START="$(date +%s)"
"$PY" tools/train_plan_a_clean_head_p1.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --teacher-dir "" \
  --contract "$CONTRACT" --contract-mode smoke --train-turner off \
  --raw-score-arm marginal --shared-init "$INIT" \
  --out "$STAGE/marginal" --snapshot-dir "$STAGE/marginal/snapshots" \
  --seed 0 --steps 2 --batch-size 4 --head-lr 1e-4 --backbone-lr 1e-6 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --max-unfrozen-blocks 33 --warmup-steps 5 --grad-clip 1 \
  --nll-normalization length --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
  --lambda-nll 1 --lambda-marginal-bce 1 \
  --marginal-vjp-method exact_tangent --marginal-vjp-epsilon 1e-3 \
  --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
  --scorer resnet2d --no-early-stop --save-every 500 \
  --snapshot-every 0 --snapshot-steps "" \
  > "$STAGE/marginal.train.stdout.log" 2>&1
test -s "$STAGE/marginal/snapshots/marginal_step2.pt"
"$PY" tools/export_prior_alignment_scores.py \
  --checkpoint "$STAGE/marginal/snapshots/marginal_step2.pt" \
  --train-trajectory train_off --contract-mode smoke \
  --out "$STAGE/marginal/export_train_off" --limit 4 --resume \
  > "$STAGE/marginal.export.stdout.log" 2>&1
"$PY" tools/evaluate_raw_score_matrix.py \
  --contract "$CONTRACT" --seed 0 --steps 2 --stage "$STAGE" \
  --out "$STAGE/evaluation_v2" --limit 4 \
  > "$STAGE/evaluation_v2.stdout.log" 2>&1
"$PY" tools/verify_raw_score_matrix.py \
  --contract "$CONTRACT" --seed 0 --steps 2 --stage "$STAGE" \
  --evaluation "$STAGE/evaluation_v2" --out "$STAGE/verification_v2_smoke.json" --limit 4 \
  > "$STAGE/verification_v2.stdout.log" 2>&1
END="$(date +%s)"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/gpu_after.txt"
"$PY" - "$STAGE" "$((END-START))" <<'PY2'
import json, sys
from pathlib import Path
stage = Path(sys.argv[1])
verdict = json.loads((stage / "evaluation_v2/pilot_verdict.json").read_text())
verification = json.loads((stage / "verification_v2_smoke.json").read_text())
assert verification["status"] == "PASS"
assert verdict["decision"] == "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT"
summary = {
    "status": "PASS",
    "scope": "ENGINEERING_SMOKE_ONLY_NO_SCIENTIFIC_VERDICT",
    "elapsed_seconds_train_export_evaluate_verify": int(sys.argv[2]),
    "verification_sha256": __import__("hashlib").sha256((stage / "verification_v2_smoke.json").read_bytes()).hexdigest(),
    "decision": verdict["decision"],
    "smoke_gate_candidates_not_scientific_promotions": verdict["promoted_arms"],
    "scientifically_promoted_arms": [],
    "external_evaluation_allowed": False,
}
(stage / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY2
