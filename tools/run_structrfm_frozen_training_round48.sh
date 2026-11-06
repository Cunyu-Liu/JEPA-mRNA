#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
EMB=$ROOT/data/embeddings_structrfm_round48
while [ ! -f "$EMB/bprna_vl0.shard0of1.npz" ]; do sleep 30; done
export PYTHONPATH=$REPO/src:$REPO/tools
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$ROOT/logs/round48"
start_seed() {
  local seed=$1
  local out=$ROOT/runs/structrfm_frozen_jev_s$seed
  mkdir -p "$out"
  local resume_args=()
  if [ -f "$out/resume.pt" ]; then
    resume_args=(--resume "$out/resume.pt")
  fi
  $PY "$REPO/src/rnajepa/train_decision.py" --arm "structrfm_frozen_jev_s$seed" --out "$out" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/bprna_vl0.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" --validation-max-examples 196 \
    --embedding-dir "$EMB" --embedding-d-model 768 --scorer resnet2d --head-chunk-size 0 \
    --device cuda --steps 0 --batch-size 2 --seed "$seed" --lr 1e-4 --weight-decay 0 \
    --warmup-steps 5 --grad-clip 1 --log-every 25 --save-every 500 \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 --nll-normalization sum \
    "${resume_args[@]}" > "$ROOT/logs/round48/structrfm_frozen_jev_s${seed}.log" 2>&1
}
start_seed 0 & pid0=$!
start_seed 1 & pid1=$!
set +e
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
set -e
if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
  echo "structRFM frozen seed failure: s0=$status0 s1=$status1" >&2
  exit 1
fi
start_seed 2
