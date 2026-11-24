#!/usr/bin/env bash
set -euo pipefail

REPO=/home/rna_jev_xens2_dualview_round56
PYTHON=/root/miniconda3/envs/lucaone_cu128/bin/python
export PYTHONPATH="$REPO/src:$REPO"
export CUDA_VISIBLE_DEVICES=0
export RN_CUDA_INSIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$REPO"
exec "$PYTHON" "$REPO/tools/run_gec_seed0_production_v1.py" "$@"
