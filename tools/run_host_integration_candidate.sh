#!/usr/bin/env bash
# P1 engineering-only candidate revalidation; not a pilot.
set -uo pipefail
NAME="${1:?usage: run_host_integration_candidate.sh h0|h2_seed0|h2_seed1_resume}"
case "$NAME" in
  h0) ARM=H0; SEED=0 ;;
  h2_seed0) ARM=H2; SEED=0 ;;
  h2_seed1_resume) ARM=H2; SEED=1 ;;
  *) echo "invalid name $NAME" >&2; exit 2 ;;
esac
REPO=/home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/p1
BASE="$ART/host_integration"
if [ "$NAME" = h2_seed1_resume ]; then
  OUT="$ART/host_integration_resume_fixture_v2/$NAME"
else
  OUT="$ART/host_integration_candidate_v1/$NAME"
fi
INIT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt
if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing $OUT" >&2
  exit 3
fi
mkdir -p "$OUT/snapshots"
cd "$REPO" || exit 2
sha256sum tools/train_plan_a.py src/rnajepa/dual_view.py tests/test_dual_view.py > "$OUT/source_sha256.txt"
run_step_cap() {
  local steps="$1"
  local log="$2"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" tools/train_plan_a.py \
    --data "$BASE/integration_train.jsonl" \
    --validation-data "$BASE/integration_dev.jsonl" \
    --teacher-dir "" \
    --init-checkpoint "$INIT" \
    --lambda-distill 0 \
    --out "$OUT" \
    --snapshot-dir "$OUT/snapshots" \
    --steps "$steps" \
    --save-every 1 \
    --snapshot-every 1 \
    --validation-every 1 \
    --validation-patience 1 \
    --warmup-head-steps 0 \
    --unfreeze-every 1 \
    --unfreeze-per-step 2 \
    --batch-size 2 \
    --seed "$SEED" \
    --scorer resnet2d \
    --dual-view-arm "$ARM" \
    --dual-view-log-every 1 \
    > "$log" 2>&1
}
run_step_cap 1 "$OUT/stdout_step1.log"
CODE=$?
printf '%s\n' "$CODE" > "$OUT/exit_step1"
if [ "$CODE" -ne 0 ] || [ "$NAME" != h2_seed1_resume ]; then
  exit "$CODE"
fi
"$PY" - "$OUT/resume.pt" "$OUT/legacy_resume_preparation.json" <<'PY'
import json
import os
import sys
import torch
path, report_path = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
assert int(state["step"]) == 1
sigreg_step = int(state["dual_view"]["sigreg.global_step"].item())
assert sigreg_step >= 1
assert int(state["dual_view_sigreg_step"]) == sigreg_step
state.pop("dual_view_sigreg_step")
temporary = path + ".legacy.tmp"
torch.save(state, temporary)
os.replace(temporary, path)
report = {
    "status": "LEGACY_EXPLICIT_FIELD_REMOVED_FOR_ENGINEERING_RESUME_TEST",
    "step": 1,
    "sigreg_global_step_preserved_in_state_dict": sigreg_step,
    "note": "Only the redundant explicit scalar was removed; optimizer and dual_view state were retained."
}
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report), flush=True)
PY
CODE=$?
if [ "$CODE" -ne 0 ]; then
  printf '%s\n' "$CODE" > "$OUT/exit_legacy_preparation"
  exit "$CODE"
fi
run_step_cap 2 "$OUT/stdout_resume_step2.log"
CODE=$?
printf '%s\n' "$CODE" > "$OUT/exit_resume_step2"
exit "$CODE"
