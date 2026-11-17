#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?usage: run_v2_bpfold_pilot.sh inner_outer|shuffled|turner0|energy_only}"
case "$ARM" in inner_outer|shuffled|turner0|energy_only) ;; *) echo "invalid BPfold arm" >&2; exit 2;; esac
REPO=/home/rna_jev_xens2_dualview_round56
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56
OUT="$ART/artifacts/v2/bpfold_v5/$ARM"
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
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --validation-teacher-dir /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/teacher_vl0 \
  --init-checkpoint "$INIT" \
  --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 \
  --nll-normalization sum --out "$OUT" --snapshot-dir "$OUT/snapshots" \
  --steps 500 --save-every 25 --snapshot-every 100 \
  --validation-every 100 --validation-patience 3 --validation-min-delta 0.000001 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --batch-size 4 --seed 42 --scorer resnet2d \
  --bpfold-motif "$ARM"
"$PY" tools/evaluate_single_checkpoint.py \
  --checkpoint "$OUT/best.pt" --data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
  --data-role model-dev --repeats 1 --output "$OUT/model_dev_eval.json"
