#!/usr/bin/env bash
set -euo pipefail
seed=${1:?seed required: 42, 2026, or 3407}
[[ "$seed" == 42 || "$seed" == 2026 || "$seed" == 3407 ]]
MODE=${E2B_MODE:-online}
[[ "$MODE" == online || "$MODE" == shuffled ]]
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
ARM=e2b
[[ "$MODE" == shuffled ]] && ARM=e2b_shuffled
OUT=$ROOT/runs/round54_${ARM}_s${seed}
mkdir -p "$OUT" "$ROOT/weights/round54/${ARM}_s${seed}" "$ROOT/logs/round54"
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev
exec /root/miniconda3/envs/lucaone/bin/python /home/rna_jev/tools/train_plan_a.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --out "$OUT" --steps 0 \
  --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --validation-teacher-dir "$ROOT/data/teacher_vl0" \
  --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
  --batch-size 4 --seed "$seed" --warmup-head-steps 1600 \
  --unfreeze-every 800 --unfreeze-per-step 2 \
  --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
  --scorer resnet2d --pair-latent-dim 64 --thermo "$MODE" \
  --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round54/${ARM}_s${seed}" \
  --gpu-reserve-gb 0
