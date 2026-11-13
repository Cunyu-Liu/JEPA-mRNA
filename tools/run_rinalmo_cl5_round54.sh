#!/usr/bin/env bash
set -euo pipefail
seed=${1:?seed required: 42, 2026, or 3407}
[[ "$seed" == 42 || "$seed" == 2026 || "$seed" == 3407 ]]
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
OUT=$ROOT/runs/round54_cl5_s${seed}
mkdir -p "$OUT" "$ROOT/weights/round54/cl5_s${seed}" "$ROOT/logs/round55"
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
  --scorer resnet2d --pair-latent-dim 64 --cl1-token-mask 0.15 \
  --cl2-infonce-weight 0.1 --cl2-temperature 0.1 --cl2-max-negatives 32 \
  --cl3-negative-map "$ROOT/data/cl0_structure_contrast_round54/cl3_negative_map.json" \
  --cl4-negative-weights "$ROOT/data/cl0_structure_contrast_round54/cl4_negative_weights.json" \
  --cl5-output-weight 0.1 \
  --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round54/cl5_s${seed}" \
  --gpu-reserve-gb 0
