#!/usr/bin/env bash
# v2: retrain only marginal with the exact forward-mode tangent VJP.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
SEED=0
STEPS=9600
STAGE="$ART/raw_score_mechanism_4arm_v1/seed0"
CONTRACT="$REPO/configs/raw_score_mechanism_4arm_v2.json"
INIT="$ART/prior_alignment_v1/init/shared_init_seed0.pt"
FAILED_ARCHIVE="$STAGE/marginal_failed_central_difference_v1"
EVALUATION="$STAGE/evaluation_v2"
VERIFICATION="$STAGE/verification_v2.json"
cd "$REPO"
git diff --quiet
git diff --cached --quiet
for ARM in frozen limited4; do
  "$PY" - "$STAGE/$ARM/run_meta.json" "$ARM" <<'PY2'
import json, sys
meta = json.load(open(sys.argv[1]))
assert meta["status"] == "completed", sys.argv[2]
assert meta["steps_completed"] == 9600 and meta["stop_reason"] == "step_cap", sys.argv[2]
PY2
done
if [[ "${SKIP_V1_ARCHIVE:-0}" != "1" ]]; then
[[ -d "$STAGE/marginal" ]] || { echo "missing failed v1 marginal directory"; exit 2; }
[[ ! -e "$FAILED_ARCHIVE" ]] || { echo "central-difference archive already exists"; exit 2; }
[[ ! -e "$EVALUATION" && ! -e "$VERIFICATION" && ! -e "$STAGE/provenance_v2.json" ]] || {
  echo "v2 evaluation/verification already exists"; exit 2; }
[[ ! -e "$STAGE/marginal/export_train_off" ]] || { echo "unexpected v1 marginal export"; exit 2; }
"$PY" - "$STAGE" "$CONTRACT" <<'PY2'
import hashlib, json, sys
from pathlib import Path
stage = Path(sys.argv[1])
contract = Path(sys.argv[2])
run = stage / "marginal"
meta = json.loads((run / "run_meta.json").read_text())
assert meta["status"] == "diverged" and meta["steps_completed"] == 537
assert meta["stop_reason"] == "GRADIENT_NORM_THRESHOLD_EXCEEDED:8848>100"
assert not (run / "snapshots/marginal_step9600.pt").exists()
identity = json.loads(contract.read_text())["contract_migration"]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert identity["previous_contract_sha256"] == sha(contract.parent / "raw_score_mechanism_4arm_v1.json")
record = {
    "status": "NUMERICAL_VJP_FAILURE_NO_SCIENTIFIC_RESULT",
    "contract_id": "raw_score_mechanism_4arm_v1",
    "last_batch_step": 537,
    "stop_reason": meta["stop_reason"],
    "resume_checkpoint_is_not_terminal_result": True,
    "run_meta_sha256": sha(run / "run_meta.json"),
    "stdout_sha256": sha(stage / "marginal.train.stdout.log"),
    "central_difference_diagnostic_sha256": sha(stage / "marginal_step537_diagnostic.json"),
    "perturbation_diagnostic_sha256": sha(stage / "marginal_step537_perturbation_diagnostic.json"),
    "exact_tangent_diagnostic_sha256": sha(stage / "marginal_step537_exact_tangent_diagnostic.json"),
    "v2_contract_id": identity and json.loads(contract.read_text())["contract_id"],
    "v2_contract_sha256": sha(contract),
    "restart": "from shared init; no optimizer resume; frozen and limited4 unchanged",
}
(stage / "marginal_failed_central_difference_v1.json").write_text(json.dumps(record, indent=2) + "\n")
PY2
mv "$STAGE/marginal" "$FAILED_ARCHIVE"
mv "$STAGE/marginal.train.stdout.log" "$STAGE/marginal_failed_central_difference_v1.train.stdout.log"
else
[[ ! -e "$STAGE/marginal" ]] || { echo "engineering failure must be archived before restart"; exit 2; }
[[ -s "$FAILED_ARCHIVE/run_meta.json" && -s "$STAGE/marginal_failed_central_difference_v1.json" ]] || {
  echo "missing immutable v1 archive"; exit 2; }
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
RUN_GIT_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA256="$(sha256sum tools/train_plan_a_clean_head_p1.py | cut -d' ' -f1)"
MARGINAL_SHA256="$(sha256sum src/rnajepa/structured_marginal.py | cut -d' ' -f1)"
CUDA_SHA256="$(sha256sum src/rnajepa/harness_cuda.py | cut -d' ' -f1)"
INIT_SHA256="$(sha256sum "$INIT" | cut -d' ' -f1)"
CONTRACT_SHA256="$(sha256sum "$CONTRACT" | cut -d' ' -f1)"
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
  --lambda-nll 1 --lambda-marginal-bce 1 \
  --marginal-vjp-method exact_tangent --marginal-vjp-epsilon 1e-3 \
  --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
  --scorer resnet2d --no-early-stop --save-every 500 \
  --snapshot-every 0 --snapshot-steps "" \
  > "$STAGE/marginal.train.stdout.log" 2>&1
test -s "$STAGE/marginal/snapshots/marginal_step9600.pt"
"$PY" tools/export_prior_alignment_scores.py \
  --checkpoint "$STAGE/marginal/snapshots/marginal_step9600.pt" \
  --train-trajectory train_off --contract-mode pilot \
  --out "$STAGE/marginal/export_train_off" --limit 0 --resume \
  > "$STAGE/marginal.export.stdout.log" 2>&1
"$PY" tools/evaluate_raw_score_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" \
  --stage "$STAGE" --out "$EVALUATION" --limit 0 \
  > "$STAGE/evaluation_v2.stdout.log" 2>&1
RUN_GIT_HEAD="$RUN_GIT_HEAD" TRAIN_SHA256="$TRAIN_SHA256" \
MARGINAL_SHA256="$MARGINAL_SHA256" CUDA_SHA256="$CUDA_SHA256" \
INIT_SHA256="$INIT_SHA256" CONTRACT_SHA256="$CONTRACT_SHA256" STAGE="$STAGE" \
"$PY" - <<'PY2'
import json, os
from pathlib import Path
stage = Path(os.environ["STAGE"])
verdict = json.loads((stage / "evaluation_v2/pilot_verdict.json").read_text())
provenance = {k: os.environ[k] for k in (
    "RUN_GIT_HEAD", "TRAIN_SHA256", "MARGINAL_SHA256", "CUDA_SHA256", "INIT_SHA256", "CONTRACT_SHA256")}
provenance.update({
    "MODE": "pilot", "STEPS": "9600", "SEED": "0",
    "CONTRACT_ID": "raw_score_mechanism_4arm_v2",
    "V1_TO_V2_SCOPE": "marginal VJP only; frozen/limited4 inherited; full control reused",
    "MARGINAL_RESTART_FROM_SHARED_INIT": True,
    "MARGINAL_OPTIMIZER_RESUME_USED": False,
    "FAILED_PARTIAL_ARCHIVES": ["marginal_failed_empty_set_v1", "marginal_failed_central_difference_v1", "marginal_failed_oom_v2_step33"],
    "verification": "independent verdict is stored separately in verification_v2.json",
    "decision": verdict["decision"], "promoted_arms": verdict["promoted_arms"],
    "external_evaluation_allowed": False, "full_control_reused_not_retrained": True,
})
(stage / "provenance_v2.json").write_text(json.dumps(provenance, indent=2) + "\n")
PY2
"$PY" tools/verify_raw_score_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" --stage "$STAGE" \
  --evaluation "$EVALUATION" --provenance "$STAGE/provenance_v2.json" \
  --out "$VERIFICATION" --limit 0 \
  > "$STAGE/verification_v2.stdout.log" 2>&1
"$PY" - "$EVALUATION/pilot_verdict.json" "$VERIFICATION" <<'PY2'
import json, sys
verdict = json.load(open(sys.argv[1]))
verification = json.load(open(sys.argv[2]))
assert verification["status"] == "PASS"
print(json.dumps({"decision": verdict["decision"], "promoted_arms": verdict["promoted_arms"],
                  "independent_verification": verification["status"],
                  "external_evaluation_allowed": False}, indent=2))
PY2
