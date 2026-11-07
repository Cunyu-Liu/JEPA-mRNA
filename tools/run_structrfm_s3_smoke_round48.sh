#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
DATA=$ROOT/data/validation/structrfm_smoke_long2.jsonl
if [ ! -f "$DATA" ]; then
  echo "missing frozen long-sequence smoke input: $DATA" >&2
  exit 1
fi
mkdir -p "$ROOT/weights/round48_s3_smoke" "$ROOT/logs/round48"
$PY "$REPO/tools/profile_structrfm_s3_round48.py" --pretrained "$ROOT/weights/structrfm_round48" \
  --lengths 128 256 500 --out "$ROOT/audits/round48/structrfm_s3_flops.json" \
  >> "$ROOT/logs/round48/structrfm_s3_flops.log" 2>&1
pids=()
for seed in 0 1; do
  out=$ROOT/runs/structrfm_s3_cuda_smoke_s$seed
  mkdir -p "$out"
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data "$DATA" --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --out "$out" --steps 2 --batch-size 2 --seed "$seed" \
    --save-every 2 --snapshot-every 2 --snapshot-dir "$ROOT/weights/round48_s3_smoke" \
    --warmup-head-steps 0 --unfreeze-every 1 --unfreeze-per-step 12 \
    --head-lr 1e-4 --backbone-lr 1e-5 --zchunk 64 \
    --scorer pairmlp --pair-latent-dim 490 \
    >> "$ROOT/logs/round48/structrfm_s3_cuda_smoke_s${seed}.log" 2>&1 &
  pids+=("$!")
done
set +e
wait "${pids[0]}"; status0=$?
wait "${pids[1]}"; status1=$?
set -e
printf 'S3 full CUDA smoke statuses: seed0=%s seed1=%s\n' "$status0" "$status1"
test "$status0" -eq 0 && test "$status1" -eq 0
$PY "$REPO/tools/eval_structrfm_full_round48.py" \
  --checkpoint "$ROOT/weights/round48_s3_smoke/structrfm_s3_cuda_smoke_s0_step2.pt" \
  --pretrained "$ROOT/weights/structrfm_round48" --data "$DATA" --repeats 3 \
  --out "$ROOT/eval_decision/structrfm_s3_smoke/s0/structrfm_smoke_long2.json" \
  >> "$ROOT/logs/round48/structrfm_s3_cuda_smoke_eval.log" 2>&1
