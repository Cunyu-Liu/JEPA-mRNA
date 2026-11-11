#!/usr/bin/env bash
set -euo pipefail
seed=${1:?seed required: 42, 2026, or 3407}
[[ "$seed" == 42 || "$seed" == 2026 || "$seed" == 3407 ]]
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
DATA=$ROOT/data/rna_complex_round49/rinalmo_d1_length_round54/1022
OUT=$ROOT/runs/round54_d1_s${seed}
mkdir -p "$OUT" "$ROOT/weights/round54/d1_s${seed}" "$ROOT/logs/round54"
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev
exec /root/miniconda3/envs/lucaone/bin/python /home/rna_jev/tools/train_plan_a.py \
  --data "$DATA/train_d1.jsonl" \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --aux-teacher-dir "$ROOT/data/rna_complex_round49/rinalmo_full_d1/teacher_complex" \
  --lazy-teacher --out "$OUT" --steps 0 \
  --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --validation-teacher-dir "$ROOT/data/teacher_vl0" \
  --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
  --batch-size 4 --seed "$seed" --warmup-head-steps 1600 \
  --unfreeze-every 800 --unfreeze-per-step 2 \
  --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
  --scorer resnet2d --pair-latent-dim 64 --save-every 500 \
  --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round54/d1_s${seed}" \
  --gpu-reserve-gb 0
