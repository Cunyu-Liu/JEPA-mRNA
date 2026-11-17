#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
BASE="$ART/artifacts/p1/host_integration"
OUT="$ART/artifacts/v2/mechanism/h2_profile"
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
INIT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt
if [ -e "$OUT" ]; then
  echo "refusing to overwrite $OUT" >&2
  exit 3
fi
mkdir -p "$OUT/snapshots"
cd "$REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" tools/train_plan_a.py \
  --data "$BASE/integration_train.jsonl" \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --init-checkpoint "$INIT" \
  --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 \
  --out "$OUT" --snapshot-dir "$OUT/snapshots" \
  --steps 1 --save-every 1 --snapshot-every 1 \
  --warmup-head-steps 0 --unfreeze-every 1 --unfreeze-per-step 2 \
  --batch-size 2 --seed 1 --scorer resnet2d \
  --dual-view-arm H2 --dual-view-log-every 1 \
  --profile-out "$OUT/full_host_profile.json"
