#!/usr/bin/env bash
set -euo pipefail

REPO=/home/rna_jev_xens2_dualview_round56
PYTHON=/root/miniconda3/envs/lucaone_cu128/bin/python
ROOT=/teams/HoulabData_1788502323/liucunyu/foldnar/gec_smoke_v1
CONFIG="$REPO/configs/gec_four_arm_convergence_v1.json"
SHARED_INIT=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1/init/shared_init_seed0.pt
GEC_INIT="$ROOT/gec_shared_residual_init_seed0.pt"
export PYTHONPATH="$REPO/src:$REPO"
export CUDA_VISIBLE_DEVICES=0
export RN_CUDA_INSIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT"
printf '%s\n' "python=$PYTHON" "repo=$REPO" "config=$CONFIG" "device=cuda:0" "RN_CUDA_INSIDE=1" > "$ROOT/launch_environment_v1.txt"
if [[ ! -f "$GEC_INIT" ]]; then
  "$PYTHON" "$REPO/tools/init_gec_residual_v1.py" \
    --shared-init "$SHARED_INIT" --gec-config "$CONFIG" --out "$GEC_INIT" --seed 0 \
    2>&1 | tee "$ROOT/init_gec_residual_v1.log"
fi

for ARM in matched_control ensemble_context destroy_interval_shuffle independent_context; do
  OUT="$ROOT/$ARM"
  mkdir -p "$OUT"
  if [[ -f "$OUT/run_meta.json" ]] && "$PYTHON" - "$OUT/run_meta.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
raise SystemExit(0 if row.get("status") == "engineering_smoke_completed" and row.get("optimizer_steps") == 2 else 1)
PY
  then
    printf 'Skipping already completed arm: %s\n' "$ARM"
    continue
  fi
  "$PYTHON" "$REPO/tools/train_plan_a_gec_convergence_v1.py" \
    --gec-config "$CONFIG" --arm "$ARM" --out "$OUT" \
    --shared-init "$SHARED_INIT" --gec-init "$GEC_INIT" \
    --device cuda:0 --engineering-smoke --steps 2 \
    --train-record-limit 8 --validation-record-limit 2 \
    --validation-every 0 --save-every 1 \
    2>&1 | tee -a "$OUT.command.log"
done

"$PYTHON" "$REPO/tools/verify_gec_smoke_v1.py" --root "$ROOT" --repo "$REPO" \
  2>&1 | tee "$ROOT/verify_gec_smoke_v1.log"
