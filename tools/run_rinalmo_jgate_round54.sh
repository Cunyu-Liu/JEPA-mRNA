#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
OUT=$ROOT/runs/round54_jgate_s42_cuda_r1
mkdir -p "$OUT" "$ROOT/weights/round54/jgate_s42_cuda" "$ROOT/logs/round54"
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev
exec /root/miniconda3/envs/lucaone/bin/python /home/rna_jev/tools/train_plan_a.py \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --out "$OUT" --steps 1 \
  --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --validation-teacher-dir "$ROOT/data/teacher_vl0" --validation-every 500 \
  --batch-size 1 --seed 42 --warmup-head-steps 0 \
  --unfreeze-every 1 --unfreeze-per-step 33 \
  --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
  --scorer pairjepa --pair-latent-dim 490 --lambda-jepa 0.1 \
  --jepa-target ema --jepa-context structured_block --jepa-mask-prob 0.15 \
  --save-every 1 --snapshot-every 2000 \
  --snapshot-dir "$ROOT/weights/round54/jgate_s42_cuda" --gpu-reserve-gb 0
