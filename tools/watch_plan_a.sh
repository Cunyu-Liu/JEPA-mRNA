#!/bin/bash
# Watch the Plan-A training run; keep it alive and evaluate when done.
#
# Robustness loop: the full-unfreeze phase peaks at ~14.1GB and shares a
# 3g.20gb MIG slice with other users' processes that can grow; an OOM kill
# would otherwise strand the run. The trainer checkpoints resume.pt every
# 500 steps and auto-resumes, so the watcher relaunches it (through the same
# wait-for-device logic as the original launcher) and keeps watching, up to
# MAX_RESTARTS times. On DONE it runs the final evaluation.
set -u

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools:/home/cunyuliu/rna-jepa/eval
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
LOG=$D/runs/plana_giga_s0.log
TRAIN_OUT=$D/runs/plana_giga_s0
PID_FILE=$D/runs/plana_giga_s0.launch_pid

CANDIDATES=(
  "MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
  "MIG-10b9b777-a776-56de-9f7c-efde6c584f71"
  "GPU-2"
  "GPU-0"
  "GPU-3"
)
NEED_GB=15
POLL=120
MAX_RESTARTS=8
MAX_WAIT_HOURS=24

free_gb() {
  CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null
}

wait_device() {
  local waited=0
  while true; do
    for c in "${CANDIDATES[@]}"; do
      local fb
      fb=$(free_gb "$c")
      if [ -n "$fb" ]; then
        local ok
        ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
        if [ "$ok" = "1" ]; then
          echo "$c"
          return 0
        fi
      fi
    done
    if [ "$waited" -ge $((MAX_WAIT_HOURS * 3600 / POLL)) ]; then
      return 1
    fi
    sleep "$POLL"
    waited=$((waited + 1))
  done
}

restarts=0
while true; do
  TRAIN_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -z "$TRAIN_PID" ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    if grep -q "\[plan-a\] DONE" "$LOG" 2>/dev/null; then
      echo "[watch_plan_a] training DONE $(date '+%T')"
      break
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "[watch_plan_a] FATAL: exceeded $MAX_RESTARTS restarts"
      tail -20 "$LOG"
      exit 1
    fi
    echo "[watch_plan_a] trainer died without DONE (restart $restarts); tail:"
    tail -5 "$LOG"
    DEV=$(wait_device) || { echo "[watch_plan_a] no device; giving up"; exit 1; }
    echo "[watch_plan_a] relaunching on $DEV $(date '+%T')"
    cd "$REPO" || exit 1
    CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
      --out "$TRAIN_OUT" \
      --data "$D/ss_data/jsonl/bprna_tr0.jsonl" \
      --teacher-dir "$D/ss_data/teacher/bprna_tr0" \
      --steps 20000 --batch-size 4 \
      --head-lr 1e-4 --backbone-lr 1e-5 \
      --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
      --save-every 500 --snapshot-every 2000 \
      >> "$LOG" 2>&1 < /dev/null &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    echo "$DEV" > "$D/runs/plana_giga_s0.launch_dev"
    echo "[watch_plan_a] relaunched pid=$NEW_PID on $DEV"
    restarts=$((restarts + 1))
    sleep 60
    continue
  fi
  sleep 120
done

CKPT=$D/ckpts/plana_giga_s0_step20000.pt
if [ ! -f "$CKPT" ]; then
  echo "[watch_plan_a] FATAL: no final checkpoint at $CKPT"
  exit 1
fi

echo "[watch_plan_a] evaluating final checkpoint on TS0 + bpRNA-new"
DEV=$(wait_device) || DEV=MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585
CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" /home/cunyuliu/rna-jepa/tools/eval_plan_a.py \
  --checkpoint "$CKPT" \
  --out $D/eval_decision/plana_giga_s0_step20000/result.json \
  > $D/runs/plana_giga_s0_eval.log 2>&1

echo "[watch_plan_a] evaluation done $(date '+%T'); result:"
tail -6 $D/runs/plana_giga_s0_eval.log
