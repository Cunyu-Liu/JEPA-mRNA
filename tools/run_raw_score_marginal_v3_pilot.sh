#!/usr/bin/env bash
# Archive failed v2 exact tangent, then train/evaluate/verify the v3 seed0 pilot.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
STAGE="$ART/raw_score_mechanism_4arm_v1/seed0"
CONTRACT="$REPO/configs/raw_score_mechanism_4arm_v3.json"
SMOKE="$ART/raw_score_mechanism_v3_smoke/seed0/smoke_summary.json"
STRESS="$STAGE/marginal_v3_numerical_stress.json"
FAILED_ARCHIVE="$STAGE/marginal_failed_nonfinite_tangent_v2"
FAILED_RECORD="$STAGE/marginal_failed_nonfinite_tangent_v2.json"
FAILED_STDOUT="$STAGE/marginal_failed_nonfinite_tangent_v2.train.stdout.log"
WAIT_LOG="$STAGE/marginal_v3_resource_wait.log"
MIN_FREE_MB=20000
cd "$REPO"
git diff --quiet
git diff --cached --quiet
[[ -d "$STAGE/frozen" && -d "$STAGE/limited4" ]] || { echo "missing inherited arms"; exit 2; }
[[ ! -e "$STAGE/evaluation_v3" && ! -e "$STAGE/verification_v3.json" ]] || { echo "v3 evaluation already exists"; exit 2; }
[[ -f "$SMOKE" && -f "$STRESS" && -f "$FAILED_RECORD" && -d "$STAGE/marginal" ]] || { echo "missing smoke, stress, failure record, or failed v2 run"; exit 2; }
"$PY" - "$STAGE" "$CONTRACT" "$SMOKE" "$STRESS" "$FAILED_RECORD" <<'PY'
import json, sys
from pathlib import Path
stage = Path(sys.argv[1])
contract = json.loads(Path(sys.argv[2]).read_text())
smoke = json.loads(Path(sys.argv[3]).read_text())
stress = json.loads(Path(sys.argv[4]).read_text())
failure = json.loads(Path(sys.argv[5]).read_text())
assert contract["contract_id"] == "raw_score_mechanism_4arm_v3"
assert smoke["status"] == "PASS" and smoke["scientifically_promoted_arms"] == []
assert stress["status"] == "PASS" and "NO_SCIENTIFIC_RESULT" in stress["scope"]
assert failure["last_batch_step"] == 9198
assert failure["status"] == "NUMERICAL_NONFINITE_TANGENT_NO_SCIENTIFIC_RESULT"
assert failure["resume_checkpoint_is_not_terminal"] is True
for arm, blocks in (("frozen", 0), ("limited4", 4)):
    meta = json.loads((stage / arm / "run_meta.json").read_text())
    assert meta["status"] == "completed" and meta["stop_reason"] == "step_cap" and meta["batch_step"] == 9600
    assert meta["unfrozen_blocks"] == blocks
    assert (stage / arm / "export_train_off" / "scores_manifest.json").exists()
failed_meta = json.loads((stage / "marginal/run_meta.json").read_text())
assert failed_meta["batch_step"] == 9198
assert not (stage / "marginal/snapshots/marginal_step9600.pt").exists()
assert not (stage / "marginal/export_train_off").exists()
assert "non-finite exact structured-marginal VJP" in (stage / "marginal.train.stdout.log").read_text()
assert (stage / "marginal/resume.pt").exists()
PY
[[ ! -e "$FAILED_ARCHIVE" && ! -e "$FAILED_STDOUT" ]] || { echo "v2 failure archive already exists"; exit 2; }
mv "$STAGE/marginal" "$FAILED_ARCHIVE"
mv "$STAGE/marginal.train.stdout.log" "$FAILED_STDOUT"
cp "$CONTRACT" "$STAGE/contract.v3.executed.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
export CUDA_VISIBLE_DEVICES=0
while true; do
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  NOW="$(date -Is)"
  echo "$NOW free_mb=$FREE_MB required_mb=$MIN_FREE_MB" | tee -a "$WAIT_LOG"
  if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB >= MIN_FREE_MB )); then
    break
  fi
  sleep 300
done
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/marginal_v3_gpu_before.txt"
START="$(date +%s)"
"$PY" tools/train_plan_a_clean_head_p1.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --teacher-dir "" \
  --contract "$CONTRACT" --contract-mode pilot --train-turner off \
  --raw-score-arm marginal --shared-init "$ART/prior_alignment_v1/init/shared_init_seed0.pt" \
  --out "$STAGE/marginal" --snapshot-dir "$STAGE/marginal/snapshots" \
  --seed 0 --steps 9600 --batch-size 4 --head-lr 1e-4 --backbone-lr 1e-6 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --max-unfrozen-blocks 33 --warmup-steps 5 --grad-clip 1 \
  --nll-normalization length --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
  --lambda-nll 1 --lambda-marginal-bce 1 \
  --marginal-vjp-method exact_tangent --marginal-vjp-epsilon 1e-3 \
  --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
  --scorer resnet2d --no-early-stop --save-every 500 \
  --snapshot-every 0 --snapshot-steps 9600 \
  > "$STAGE/marginal_v3.train.stdout.log" 2>&1
test -s "$STAGE/marginal/snapshots/marginal_step9600.pt"
"$PY" tools/export_prior_alignment_scores.py \
  --checkpoint "$STAGE/marginal/snapshots/marginal_step9600.pt" \
  --train-trajectory train_off --contract-mode pilot \
  --out "$STAGE/marginal/export_train_off" \
  > "$STAGE/marginal_v3.export.stdout.log" 2>&1
"$PY" tools/evaluate_raw_score_matrix.py \
  --contract "$CONTRACT" --seed 0 --steps 9600 --stage "$STAGE" --out "$STAGE/evaluation_v3" \
  > "$STAGE/evaluation_v3.stdout.log" 2>&1
"$PY" tools/verify_raw_score_matrix.py \
  --contract "$CONTRACT" --seed 0 --steps 9600 --stage "$STAGE" \
  --evaluation "$STAGE/evaluation_v3" --out "$STAGE/verification_v3.json" \
  > "$STAGE/verification_v3.stdout.log" 2>&1
END="$(date +%s)"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/marginal_v3_gpu_after.txt"
"$PY" - "$STAGE" "$CONTRACT" "$((END-START))" "$STRESS" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
stage = Path(sys.argv[1])
contract = Path(sys.argv[2])
verdict = json.loads((stage / "evaluation_v3/pilot_verdict.json").read_text())
verification = json.loads((stage / "verification_v3.json").read_text())
assert verification["status"] == "PASS"
provenance = {
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "numerical_stress_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "marginal_restarted_from_shared_init_not_v2_resume": True,
    "failed_v2_archive": "marginal_failed_nonfinite_tangent_v2",
    "elapsed_seconds_marginal_train_export_evaluate_verify": int(sys.argv[3]),
    "evaluation_sha256": {
        name: hashlib.sha256((stage / "evaluation_v3" / name).read_bytes()).hexdigest()
        for name in ["arm_metrics.json", "pairwise_comparisons.json", "pilot_verdict.json"]
    },
    "verification_sha256": hashlib.sha256((stage / "verification_v3.json").read_bytes()).hexdigest(),
}
(stage / "provenance_v3.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"verification": verification["status"], "verdict": verdict, "provenance": provenance}, indent=2))
PY
