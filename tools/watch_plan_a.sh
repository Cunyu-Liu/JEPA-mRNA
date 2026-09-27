#!/bin/bash
# Watch the Plan-A training run; when it completes, evaluate TS0 + bpRNA-new
# with the standard protocol (eval_plan_a.py) and write result JSONs.
#
# Gating: only the final snapshot (step 20000) is evaluated -- intermediate
# snapshots exist for the step-trend analysis but are not this arm's headline.
set -u

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools:/home/cunyuliu/rna-jepa/eval
D=/mnt/cunyuliu/rna-jepa
LOG=$D/runs/plana_giga_s0.log
TRAIN_PID=$(cat $D/runs/plana_giga_s0.launch_pid 2>/dev/null || echo "")

echo "[watch_plan_a] watching pid=$TRAIN_PID log=$LOG"

while true; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    if grep -q "DONE" "$LOG" 2>/dev/null; then
      echo "[watch_plan_a] training DONE detected $(date '+%T')"
      break
    fi
    echo "[watch_plan_a] FATAL: pid $TRAIN_PID exited without DONE"
    tail -20 "$LOG"
    exit 1
  fi
  if grep -q "\[plan-a\] DONE" "$LOG" 2>/dev/null; then
    echo "[watch_plan_a] DONE in log while process still alive; waiting for exit"
  fi
  sleep 120
done

CKPT=$D/ckpts/plana_giga_s0_step20000.pt
if [ ! -f "$CKPT" ]; then
  echo "[watch_plan_a] FATAL: no final checkpoint at $CKPT"
  exit 1
fi

echo "[watch_plan_a] evaluating final checkpoint on TS0 + bpRNA-new"
CUDA_VISIBLE_DEVICES=MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" /home/cunyuliu/rna-jepa/tools/eval_plan_a.py \
  --checkpoint "$CKPT" \
  --out $D/eval_decision/plana_giga_s0_step20000/result.json \
  > $D/runs/plana_giga_s0_eval.log 2>&1

echo "[watch_plan_a] evaluation done; result:"
tail -5 $D/runs/plana_giga_s0_eval.log
