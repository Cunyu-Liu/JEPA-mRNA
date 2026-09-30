#!/bin/bash
# Plan-A third seed (s2): resolve the OOD seed-swing with a third point.
#
# Why (2026-10-01, 14.98): the two plana seeds' OOD swing (bpRNA-new
# 0.4302 -> 0.3763) is of the same order as the ID gain itself; two points
# cannot support any central-tendency statement, and the draft can only
# report them side by side. A third seed at identical protocol (seed 2)
# either tightens the spread (mean becomes quotable) or confirms the
# noise, both publishable. GPU7's two 3g.20gb MIG slices are free
# (16.9/20.3GB at launch; plana peak ~14GB fits).
set -u

# single-instance guard, fd never leaks to children (9>&- at launch)
exec 9>/mnt/cunyuliu/rna-jepa/runs/.watch_plan_a_s2.lock
flock -n 9 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools:/home/cunyuliu/rna-jepa/eval
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
LOG=$D/runs/plana_giga_s2.log
TRAIN_OUT=$D/runs/plana_giga_s2
PID_FILE=$D/runs/plana_giga_s2.launch_pid
STEPS=20000

CANDIDATES=(
  "MIG-10b9b777-a776-56de-9f7c-efde6c584f71"
  "MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
)
NEED_GB=15
POLL=120
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
    echo "[watch_s2] no device yet (poll $waited) $(date '+%T')" >&2
    sleep "$POLL"
    waited=$((waited + 1))
  done
}

restarts=0
while true; do
  TRAIN_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -z "$TRAIN_PID" ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    if grep -q "\[plan-a\] DONE" "$LOG" 2>/dev/null; then
      echo "[watch_s2] training DONE $(date '+%T')"
      break
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "[watch_s2] FATAL: exceeded $MAX_RESTARTS restarts"
      tail -20 "$LOG"; exit 1
    fi
    echo "[watch_s2] trainer died without DONE (restart $restarts); tail:"
    if pgrep -f "train_plan_a.py.*--out $TRAIN_OUT" > /dev/null; then
      echo "[watch_s2] trainer alive via pgrep (launcher won the race); watching it"
      sleep 120; continue
    fi
    tail -5 "$LOG"
    DEV=$(wait_device) || { echo "[watch_s2] no device; giving up"; exit 1; }
    echo "[watch_s2] relaunching on $DEV $(date '+%T')"
    cd "$REPO" || exit 1
    CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
      --out "$TRAIN_OUT" \
      --data "$D/ss_data/jsonl/bprna_tr0.jsonl" \
      --teacher-dir "$D/ss_data/teacher/bprna_tr0" \
      --steps "$STEPS" --batch-size 4 \
      --head-lr 1e-4 --backbone-lr 1e-5 \
      --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
      --save-every 500 --snapshot-every 2000 --seed 2 --gpu-reserve-gb 12 \
      9>&- >> "$LOG" 2>&1 < /dev/null &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    echo "$DEV" > "$D/runs/plana_giga_s2.launch_dev"
    echo "[watch_s2] relaunched pid=$NEW_PID on $DEV"
    restarts=$((restarts + 1))
    sleep 60
    continue
  fi
  sleep 120
done

# belt-and-braces terminal status (patched trainer also writes it)
"$PY" - "$TRAIN_OUT" "$STEPS" <<'PYEOF'
import json, os, sys
p = os.path.join(sys.argv[1], "run_meta.json")
m = json.load(open(p))
m["status"] = "completed"; m["steps_completed"] = int(sys.argv[2])
json.dump(m, open(p, "w"), indent=1)
print("[watch_s2] run_meta status=completed written")
PYEOF

CKPT=$D/ckpts/plana_giga_s2_step${STEPS}.pt
if [ ! -f "$CKPT" ]; then
  echo "[watch_s2] FATAL: no final checkpoint at $CKPT"
  exit 1
fi

echo "[watch_s2] evaluating final checkpoint on TS0 + bpRNA-new"
DEV=$(wait_device) || DEV=MIG-10b9b777-a776-56de-9f7c-efde6c584f71
CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" /home/cunyuliu/rna-jepa/tools/eval_plan_a.py \
  --checkpoint "$CKPT" \
  --out $D/eval_decision/plana_giga_s2_step${STEPS}/result.json \
  > $D/runs/plana_giga_s2_eval.log 2>&1

echo "[watch_s2] evaluation done $(date '+%T'); result:"
tail -6 $D/runs/plana_giga_s2_eval.log
