#!/bin/bash
# Watch the Plan-A s1 training run; keep it alive and evaluate when done. (v2)
#
# v2 fixes over v1 (2026-09-29 6th-round handover):
#   - the trainer process launched before the train_plan_a.py patch still
#     writes its snapshots under the hardcoded name plana_giga_s0_step{N}.pt
#     (s0's originals were moved to ckpts/plana_giga_s0_snapshots_preserved/
#     before the collision); any plana_giga_s0_step*.pt newer than 15:00
#     2026-09-29 therefore belongs to the s1 run and is renamed to
#     plana_giga_s1_step{N}.pt at DONE. Relaunches use the patched trainer
#     and write the correct name directly.
#   - run_meta.json terminal status: the pre-patch trainer writes no status
#     field, which made the 10-min monitor false-alert for 37h on the s0 run
#     (ledger 14.77). The watcher writes status=completed itself after DONE.
# Everything else (flock guard, restart loop, pgrep race guard, eval) is
# unchanged from v1.
set -u

# single-instance guard: duplicate invocations (e.g. replayed ssh
# commands) must exit instead of spawning a second daemon.
# NOTE (v3): the original lockfile .watch_plan_a_s1.lock is still held by
# the currently-running trainer pid -- the v1 watcher's `exec 9>` fd leaked
# into the child it launched, and the child keeps the flock for its whole
# lifetime. A new lockfile is used here, and the relaunch command closes
# fd 9 (9>&-) so the leak cannot recur.
exec 9>/mnt/cunyuliu/rna-jepa/runs/.watch_plan_a_s1_v2.lock
flock -n 9 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools:/home/cunyuliu/rna-jepa/eval
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
LOG=$D/runs/plana_giga_s1.log
TRAIN_OUT=$D/runs/plana_giga_s1
PID_FILE=$D/runs/plana_giga_s1.launch_pid
CKPT_DIR=$D/ckpts
# anything plana_giga_s0_step*.pt newer than this is the s1 run's output
# (s1 first launched 2026-09-29 15:21; s0 originals are all from 09-27/28
# and were moved to the preserved dir at 16:11)
S1_EPOCH="2026-09-29 15:00"

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
MAX_WAIT_HOURS=48

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
    echo "[watch_plan_a_s1] no device yet (poll $waited) $(date '+%T')" >&2
    sleep "$POLL"
    waited=$((waited + 1))
  done
}

restarts=0
while true; do
  TRAIN_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -z "$TRAIN_PID" ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    if grep -q "\[plan-a\] DONE" "$LOG" 2>/dev/null; then
      echo "[watch_plan_a_s1] training DONE $(date '+%T')"
      break
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "[watch_plan_a_s1] FATAL: exceeded $MAX_RESTARTS restarts"
      tail -20 "$LOG"
      exit 1
    fi
    echo "[watch_plan_a_s1] trainer died without DONE (restart $restarts); tail:"
    if pgrep -f "train_plan_a.py.*--out $TRAIN_OUT" > /dev/null; then
      echo "[watch_plan_a_s1] trainer alive via pgrep (launcher won the race); watching it"
      sleep 120
      continue
    fi
    tail -5 "$LOG"
    DEV=$(wait_device) || { echo "[watch_plan_a_s1] no device; giving up"; exit 1; }
    echo "[watch_plan_a_s1] relaunching on $DEV $(date '+%T')"
    cd "$REPO" || exit 1
    CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
      --out "$TRAIN_OUT" \
      --data "$D/ss_data/jsonl/bprna_tr0.jsonl" \
      --teacher-dir "$D/ss_data/teacher/bprna_tr0" \
      --steps 20000 --batch-size 4 \
      --head-lr 1e-4 --backbone-lr 1e-5 \
      --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
      --save-every 500 --snapshot-every 2000 --seed 1 --gpu-reserve-gb 12 \
      9>&- >> "$LOG" 2>&1 < /dev/null &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    echo "$DEV" > "$D/runs/plana_giga_s1.launch_dev"
    echo "[watch_plan_a_s1] relaunched pid=$NEW_PID on $DEV"
    restarts=$((restarts + 1))
    sleep 60
    continue
  fi
  sleep 120
done

# --- resolve the final checkpoint (patched trainer writes s1 name; the
# pre-patch process that is still running writes the s0 name) ---
FINAL=$CKPT_DIR/plana_giga_s1_step20000.pt
if [ ! -f "$FINAL" ]; then
  OLD=$CKPT_DIR/plana_giga_s0_step20000.pt
  if [ -f "$OLD" ] && find "$OLD" -newermt "$S1_EPOCH" | grep -q .; then
    mv "$OLD" "$FINAL"
    echo "[watch_plan_a_s1] renamed old-code final snapshot -> plana_giga_s1_step20000.pt"
  fi
fi
if [ ! -f "$FINAL" ]; then
  echo "[watch_plan_a_s1] FATAL: no final checkpoint at $FINAL"
  exit 1
fi

# rename any remaining old-code intermediate snapshots belonging to s1
for f in $(find "$CKPT_DIR" -maxdepth 1 -name "plana_giga_s0_step*.pt" -newermt "$S1_EPOCH"); do
  n=$(basename "$f" | sed 's/^plana_giga_s0_/plana_giga_s1_/')
  mv "$f" "$CKPT_DIR/$n"
  echo "[watch_plan_a_s1] renamed $(basename "$f") -> $n"
done

# terminal status for the 10-min monitor (lesson 14.77)
"$PY" - "$TRAIN_OUT" <<'PYEOF'
import json, os, sys
p = os.path.join(sys.argv[1], "run_meta.json")
m = json.load(open(p))
m["status"] = "completed"
m["steps_completed"] = 20000
json.dump(m, open(p, "w"), indent=1)
print("[watch_plan_a_s1] run_meta status=completed written")
PYEOF

echo "[watch_plan_a_s1] evaluating final checkpoint on TS0 + bpRNA-new"
DEV=$(wait_device) || DEV=MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585
CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" /home/cunyuliu/rna-jepa/tools/eval_plan_a.py \
  --checkpoint "$FINAL" \
  --out $D/eval_decision/plana_giga_s1_step20000/result.json \
  > $D/runs/plana_giga_s1_eval.log 2>&1

echo "[watch_plan_a_s1] evaluation done $(date '+%T'); result:"
tail -6 $D/runs/plana_giga_s1_eval.log
