#!/bin/bash
# Plan-A: gradual unfreezing of the RiNALMo-giga backbone.
#
# Control arm: rinalmo_r2d_b4_s0 (frozen giga + resnet2d head, TS0 0.6629 /
# bpRNA-new 0.5010). Single variable: backbone adaptation (in-loop training,
# top-down unfreeze 2 blocks / 800 steps after 1600 head-warmup steps,
# layer-wise LR backbone 1e-5 / head 1e-4). Everything else matches the control
# arm's protocol exactly -- see tools/train_plan_a.py's docstring.
#
# GPU policy: this arm needs ~13-15GB (650M backbone + AdamW + checkpointed
# activations). The two 3g.20gb MIG slices on GPU 7 and the full cards are
# all contended by other arms of this project and other users. The launcher
# therefore waits for any one of a set of candidate devices to free up
# (16GB threshold), then launches under setsid nohup and detaches.
set -u

# single-instance guard: duplicate invocations (e.g. replayed ssh
# commands) must exit instead of spawning a second daemon
exec 9>/mnt/cunyuliu/rna-jepa/runs/.launch_plan_a_s1.lock
flock -n 9 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa

OUT=$D/runs/plana_giga_s1
LOG=$D/runs/plana_giga_s1.log
STEPS=20000

# candidate devices, in order of preference: full card 2, the two 3g.20gb MIG
# slices (by UUID so CUDA_VISIBLE_DEVICES lands on the 20GB instance, not a
# 4.75GB 1g.5gb slice by index), then the other full cards
CANDIDATES=(
  "GPU-2"
  "MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
  "MIG-10b9b777-a776-56de-9f7c-efde6c584f71"
  "GPU-0"
  "GPU-3"
)
NEED_GB=16
POLL=120
MAX_WAIT_HOURS=96

free_gb() {
  local dev="$1"
  CUDA_VISIBLE_DEVICES="$dev" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null
}

echo "[launch_plan_a_s1] waiting for a device with >= ${NEED_GB}GB free among: ${CANDIDATES[*]}"

waited=0
DEV=""
while true; do
  for c in "${CANDIDATES[@]}"; do
    fb=$(free_gb "$c")
    if [ -n "$fb" ]; then
      ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
      if [ "$ok" = "1" ]; then
        DEV="$c"
        break 2
      fi
    fi
  done
  if [ "$waited" -ge $((MAX_WAIT_HOURS * 3600 / POLL)) ]; then
    echo "[launch_plan_a_s1] FATAL: no device freed up after ${MAX_WAIT_HOURS}h; giving up"
    exit 1
  fi
  sleep "$POLL"
  waited=$((waited + 1))
done

if pgrep -f "train_plan_a.py.*--out $OUT" > /dev/null; then
  echo "[launch_plan_a_s1] trainer already alive via pgrep (watcher won the race); exiting"
  exit 0
fi
echo "[launch_plan_a_s1] acquired $DEV (free ${fb}GB); launching"

cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES="$DEV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
  --out "$OUT" \
  --data "$D/ss_data/jsonl/bprna_tr0.jsonl" \
  --teacher-dir "$D/ss_data/teacher/bprna_tr0" \
  --steps "$STEPS" \
  --batch-size 4 \
  --head-lr 1e-4 --backbone-lr 1e-5 \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --save-every 500 --snapshot-every 2000 --seed 1 \$
  > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "[launch_plan_a_s1] launched pid=$PID on $DEV; log -> $LOG"
echo "$PID" > "$OUT.launch_pid"
echo "$DEV" > "$OUT.launch_dev"
