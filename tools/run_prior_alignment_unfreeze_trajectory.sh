#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/rna_jev_xens2_dualview_round56}"
PY="${PY:-/root/miniconda3/envs/lucaone_cu128/bin/python}"
ART="${ART:-/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1}"
cd "$REPO"

for SEED in 0 1 2; do
  STAGE="$ART/full_seed${SEED}"
  for STEP in 3200 9600 16800; do
    for TRAJECTORY in train_on train_off; do
      CHECKPOINT="$STAGE/snapshots/${TRAJECTORY}_step${STEP}.pt"
      OUT="$STAGE/trajectory/step${STEP}/raw_${TRAJECTORY}"
      echo "[trajectory] seed=$SEED step=$STEP $TRAJECTORY"
      "$PY" tools/export_prior_alignment_scores.py \
        --checkpoint "$CHECKPOINT" --train-trajectory "$TRAJECTORY" \
        --contract-mode full --out "$OUT" --device cuda --resume
    done
    "$PY" tools/analyze_prior_alignment_unfreeze_trajectory.py \
      --artifact-root "$ART" --seed "$SEED" --step "$STEP"
  done
done

"$PY" tools/analyze_prior_alignment_unfreeze_trajectory.py \
  --artifact-root "$ART" --aggregate --out "$ART/unfreeze_trajectory_analysis.json"
