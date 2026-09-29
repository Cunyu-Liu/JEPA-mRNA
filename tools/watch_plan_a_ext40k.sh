#!/bin/bash
# Watch the plana_giga_s0_ext40k run; keep it alive and evaluate when done.
#
# Same robustness pattern as watch_plan_a_s1.sh (v3): relaunch on silent
# death via resume.pt auto-resume, up to MAX_RESTARTS; on DONE evaluate.
# The trainer here was launched with the patched train_plan_a.py, so
# snapshots land under the correct name (plana_giga_s0_ext40k_step{N}.pt)
# and run_meta.json gets status=running/completed written by the trainer
# itself -- the watcher's status write is belt-and-braces (idempotent).
#
# Convergence curve: evaluates BOTH the step-30000 and step-40000
# checkpoints after DONE (s0@20000 already has its result.json), giving a
# 3-point duration axis without ever risking the training process.
set -u

# single-instance guard (fresh lockfile; fd never leaks to children: the
# relaunch command closes fd 9 via 9>&-)
exec 9>/mnt/cunyuliu/rna-jepa/runs/.watch_plan_a_ext40k.lock
flock -n 9 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools:/home/cunyuliu/rna-jepa/eval
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
LOG=$D/runs/plana_giga_s0_ext40k.log
TRAIN_OUT=$D/runs/plana_giga_s0_ext40k
PID_FILE=$D/runs/plana_giga_s0_ext40k.launch_pid
STEPS=40000

CANDIDATES=(
  "GPU-5"
  "GPU-2"
  "GPU-0"
  "GPU-3"
  "GPU-4"
  "GPU-1"
  "MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
  "MIG-10b9b777-a776-56de-9f7c-efde6c584f71"
)
NEED_GB=15
POLL=120
# 40 restarts: the cluster is in a period of extreme contention (tenants +
# q_fill daemons claim any free card within minutes, often during our
# trainer's ~6-10 min data-loading phase before it holds GPU memory), so
# failed launches are expected to repeat for hours before one lands.
MAX_RESTARTS=40
MAX_WAIT_HOURS=96

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
          echo "$c"; return 0
        fi
      fi
    done
    if [ "$waited" -ge $((MAX_WAIT_HOURS * 3600 / POLL)) ]; then
      return 1
    fi
    echo "[watch_ext40k] no device yet (poll $waited) $(date '+%T')"
    sleep "$POLL"
    waited=$((waited + 1))
  done
}

restarts=0
while true; do
  TRAIN_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -z "$TRAIN_PID" ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    if grep -q "\[plan-a\] DONE" "$LOG" 2>/dev/null; then
      echo "[watch_ext40k] training DONE $(date '+%T')"
      break
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "[watch_ext40k] FATAL: exceeded $MAX_RESTARTS restarts"
      tail -20 "$LOG"; exit 1
    fi
    echo "[watch_ext40k] trainer died without DONE (restart $restarts); tail:"
    if pgrep -f "train_plan_a.py.*--out $TRAIN_OUT" > /dev/null; then
      echo "[watch_ext40k] trainer alive via pgrep (launcher won the race); watching it"
      sleep 120; continue
    fi
    tail -5 "$LOG"
    DEV=$(wait_device) || { echo "[watch_ext40k] no device; giving up"; exit 1; }
    echo "[watch_ext40k] relaunching on $DEV $(date '+%T')"
    cd "$REPO" || exit 1
    CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
      --out "$TRAIN_OUT" \
      --data "$D/ss_data/jsonl/bprna_tr0.jsonl" \
      --teacher-dir "$D/ss_data/teacher/bprna_tr0" \
      --steps "$STEPS" --batch-size 4 \
      --head-lr 1e-4 --backbone-lr 1e-5 \
      --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
      --save-every 500 --snapshot-every 2000 --seed 0 \
      9>&- >> "$LOG" 2>&1 < /dev/null &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    echo "$DEV" > "$D/runs/plana_giga_s0_ext40k.launch_dev"
    echo "[watch_ext40k] relaunched pid=$NEW_PID on $DEV"
    restarts=$((restarts + 1))
    sleep 60
    continue
  fi
  sleep 120
done

# belt-and-braces terminal status (patched trainer already writes it)
"$PY" - "$TRAIN_OUT" "$STEPS" <<'PYEOF'
import json, os, sys
p = os.path.join(sys.argv[1], "run_meta.json")
m = json.load(open(p))
m["status"] = "completed"; m["steps_completed"] = int(sys.argv[2])
json.dump(m, open(p, "w"), indent=1)
print("[watch_ext40k] run_meta status=completed written")
PYEOF

# convergence curve: evaluate step-30000 then step-40000 (s0@20000 exists)
for STEP in 30000 40000; do
  CKPT=$D/ckpts/plana_giga_s0_ext40k_step${STEP}.pt
  if [ ! -f "$CKPT" ]; then
    echo "[watch_ext40k] NOTE: no snapshot at step $STEP (skipping its eval)"
    continue
  fi
  echo "[watch_ext40k] evaluating step $STEP on TS0 + bpRNA-new"
  DEV=$(wait_device) || DEV=GPU-5
  CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" /home/cunyuliu/rna-jepa/tools/eval_plan_a.py \
    --checkpoint "$CKPT" \
    --out $D/eval_decision/plana_giga_s0_ext40k_step${STEP}/result.json \
    > $D/runs/plana_giga_s0_ext40k_eval_${STEP}.log 2>&1
  echo "[watch_ext40k] step $STEP eval done $(date '+%T'); tail:"
  tail -4 $D/runs/plana_giga_s0_ext40k_eval_${STEP}.log
done

echo "[watch_ext40k] ALL DONE $(date '+%F %T')"
