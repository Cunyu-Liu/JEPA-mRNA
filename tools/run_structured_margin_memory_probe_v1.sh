#!/usr/bin/env bash
set -euo pipefail

REPO=/home/rna_jev_margin_candidate
PYTHON=/root/miniconda3/envs/lucaone_cu128/bin/python
ROOT=/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_memory_probe_v1

export PYTHONPATH="$REPO/src:$REPO"
export CUDA_VISIBLE_DEVICES=0
export RN_CUDA_INSIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT"
printf '%s\n' \
  "python=$PYTHON" \
  "repo=$REPO" \
  "device=cuda:0" \
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "RN_CUDA_INSIDE=$RN_CUDA_INSIDE" \
  "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG" \
  "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" \
  "min_free_gib=24" \
  > "$ROOT/launch_environment_v1.txt"

"$PYTHON" "$REPO/tools/probe_structured_margin_memory_v1.py" \
  --gec-config "$REPO/configs/gec_structured_margin_convergence_v1.json" \
  --arm ensemble_context --device cuda:0 --out "$ROOT" --min-free-gib 24 --zchunk 64 \
  2>&1 | tee "$ROOT/probe_command.log"
