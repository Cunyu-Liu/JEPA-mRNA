#!/usr/bin/env bash
# Archive the step-33 CUDA OOM, then restart only after safe free memory is available.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
STAGE="$ART/raw_score_mechanism_4arm_v1/seed0"
FAILED_ARCHIVE="$STAGE/marginal_failed_oom_v2_step33"
MIN_FREE_MB=16000
cd "$REPO"
git diff --quiet
git diff --cached --quiet
[[ -d "$STAGE/marginal" && ! -e "$FAILED_ARCHIVE" ]] || {
  echo "missing unarchived OOM run or archive already exists"; exit 2; }
[[ ! -e "$STAGE/evaluation_v2" && ! -e "$STAGE/verification_v2.json" ]] || {
  echo "formal evaluation already exists"; exit 2; }
"$PY" - "$STAGE" <<'PY2'
import hashlib, json, sys
from pathlib import Path
stage = Path(sys.argv[1])
run = stage / "marginal"
meta = json.loads((run / "run_meta.json").read_text())
assert meta["status"] == "running", "uncaught OOM leaves status=running"
assert meta["objective"]["marginal_vjp_method"] == "exact_tangent"
assert not (run / "resume.pt").exists(), "no valid optimizer resume may be claimed"
assert not (run / "snapshots/marginal_step9600.pt").exists()
batches = [json.loads(x) for x in (run / "batch_order.jsonl").read_text().splitlines() if x.strip()]
losses = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines() if x.strip()]
log = stage / "marginal.train.stdout.log"
assert "torch.OutOfMemoryError: CUDA out of memory" in log.read_text()
assert batches[-1]["step"] == 33 and losses[-1]["step"] == 25
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "status": "ENGINEERING_CUDA_OOM_NO_SCIENTIFIC_RESULT",
    "contract_id": "raw_score_mechanism_4arm_v2",
    "last_batch_step": batches[-1]["step"],
    "last_logged_step": losses[-1]["step"],
    "last_logged_peak_gb": losses[-1]["peak_gb"],
    "resume_checkpoint_written": False,
    "optimizer_resume_used_on_restart": False,
    "run_meta_sha256": sha(run / "run_meta.json"),
    "stdout_sha256": sha(log),
    "cause_scope": "GPU shared with other active RNA fine-tuning jobs; no process was terminated and no scientific/stopping parameter was changed",
    "restart": "same shared init after at least 16000 MiB GPU memory is free",
}
(stage / "marginal_failed_oom_v2_step33.json").write_text(json.dumps(record, indent=2) + "\n")
PY2
mv "$STAGE/marginal" "$FAILED_ARCHIVE"
mv "$STAGE/marginal.train.stdout.log" "$STAGE/marginal_failed_oom_v2_step33.train.stdout.log"
WAIT_LOG="$STAGE/marginal_v2_resource_wait.log"
while true; do
  FREE_MB="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  printf '%s free_mb=%s required_mb=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FREE_MB" "$MIN_FREE_MB" >> "$WAIT_LOG"
  if (( FREE_MB >= MIN_FREE_MB )); then
    break
  fi
  sleep 60
done
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/marginal_v2_restart_gpu_before.txt"
export SKIP_V1_ARCHIVE=1
exec bash "$REPO/tools/run_raw_score_marginal_v2_pilot.sh"
