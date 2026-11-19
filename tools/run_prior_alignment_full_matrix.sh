#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/rna_jev_xens2_dualview_round56}"
PY="${PY:-/root/miniconda3/envs/lucaone_cu128/bin/python}"
ART="${ART:-/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/prior_alignment_v1}"
cd "$REPO"
export PRIOR_ALIGNMENT_RUN_GIT_HEAD="$(git rev-parse HEAD)"

for SEED in 0 1 2; do
  echo "[full-matrix] starting seed ${SEED} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash tools/run_prior_alignment_stage.sh full "$SEED"
  echo "[full-matrix] completed seed ${SEED} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

"$PY" tools/verify_prior_alignment_matrix.py \
  --artifact-root "$ART" --out "$ART/full_matrix_verdict.json"
echo "[full-matrix] frozen terminal verdict written to $ART/full_matrix_verdict.json"
