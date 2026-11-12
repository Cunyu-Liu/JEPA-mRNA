#!/usr/bin/env bash
set -euo pipefail
mode=${1:-motif}
[[ "$mode" == motif || "$mode" == shuffled ]]
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
OUT=$ROOT/runs/round54_e2a_${mode}_smoke_s42
mkdir -p "$OUT" "$ROOT/weights/round54/e2a_${mode}_smoke_s42" "$ROOT/logs/round54"
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev
exec /root/miniconda3/envs/lucaone/bin/python /home/rna_jev/tools/train_plan_a.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --out "$OUT" --steps 2 \
  --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --validation-teacher-dir "$ROOT/data/teacher_vl0" \
  --validation-every 2 --validation-patience 6 --validation-min-delta 1e-6 \
  --batch-size 4 --seed 42 --warmup-head-steps 1600 \
  --unfreeze-every 800 --unfreeze-per-step 2 \
  --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
  --scorer resnet2d --pair-latent-dim 64 --motif "$mode" \
  --save-every 2 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round54/e2a_${mode}_smoke_s42" \
  --gpu-reserve-gb 0
