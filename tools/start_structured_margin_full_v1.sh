#!/usr/bin/env bash
set -euo pipefail

REPO=/home/rna_jev_margin_candidate
PYTHON=/root/miniconda3/envs/lucaone_cu128/bin/python
ROOT=/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_full_v1/generation_01

export PYTHONPATH="$REPO/src:$REPO"
export CUDA_VISIBLE_DEVICES=0
export RN_CUDA_INSIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT"
if [[ -f "$ROOT/scheduler.pid" ]]; then
  PID=$(cat "$ROOT/scheduler.pid")
  if [[ -d "/proc/$PID" ]]; then
    printf 'scheduler already running pid=%s\n' "$PID"
    exit 0
  fi
fi

nohup "$PYTHON" "$REPO/tools/run_structured_margin_full_v1.py" \
  --gec-config "$REPO/configs/gec_structured_margin_convergence_v1.json" \
  >> "$ROOT/scheduler_stdout.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ROOT/scheduler.pid"
printf 'scheduler_started pid=%s root=%s\n' "$PID" "$ROOT"
