#!/bin/bash
# Plan-B second seed: rinalmo_r2d_b4_s1 (resnet2d scorer, frozen RiNALMo-giga).
# Protocol identical to rinalmo_r2d_b4_s0 / launch_plan_b.sh; ONLY the seed differs.
#
# Why this was rewritten (2026-09-30)
# -----------------------------------
# v1 launched four times on GPU 2 (16:04, 16:35, 19:47, 22:12) and OOMed at
# step ~500 every time. Cause: the frozen-embedding store takes ~10 minutes to
# load, and `wait_device` accepted a card from ONE instantaneous reading; by the
# time the trainer reached its first backward, co-tenants had consumed the
# headroom. The daemon was also killed from outside at ~22:20 (its log ends with
# no FATAL line), so a single detached process is not a durable home for this arm.
#
# Changes: (a) a card is accepted only when it holds NEED_GB across a second
# reading taken CONFIRM_GAP seconds later - the free window must outlive the
# embedding load; (b) NEED_GB 31 -> 34, sized to the measured ~25 GiB peak plus
# co-tenant drift; (c) re-armed by a cron watchdog (scripts/ensure_r2d_s1_daemon.sh)
# instead of relying on one nohup surviving.
set -u

# Lock lives on the LOCAL filesystem: flock on /mnt/cunyuliu (shared/
# network-mounted) did not serialise - a second copy started alongside the
# first without exiting, verified 2026-09-30.
exec 9>/tmp/.r2d_s1_daemon.lock
flock -n 9 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr0.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr0
EMB=$D/embeddings/rinalmo-giga
TAG=rinalmo_r2d_b4_s1
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
NEED_GB=${R2D_NEED_GB:-34}
NEED_GB_MAX=${R2D_NEED_GB_MAX:-38}
POLL=120
CONFIRM_GAP=${R2D_CONFIRM_GAP:-90}
MAX_WAIT_HOURS=240
MAX_RESTARTS=12
cd $REPO || exit 1

if [ -f "$D/eval_decision/ow_${TAG}_step20000_bprna_ts0/result.json" ] && \
   [ -f "$D/eval_decision/ow_${TAG}_step20000_bprna_new/result.json" ]; then
  echo "[r2d_s1] both evals already present; nothing to do"
  exit 0
fi

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }

last_step() {
  tail -1 "$OUT/train_log.jsonl" 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0
}

# Two-phase acceptance: pick the emptiest card, then re-measure it after
# CONFIRM_GAP seconds and require it to still clear NEED_GB.
pick_confirmed_device() {
  local best="" bestfb=0 c fb ok
  for c in 0 1 2 3 4 5; do
    fb=$(free_gb "$c")
    [ -z "$fb" ] && continue
    ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
    if [ "$ok" = "1" ]; then
      if awk -v a="$fb" -v b="$bestfb" 'BEGIN{exit !(a>b)}'; then bestfb=$fb; best=$c; fi
    fi
  done
  [ -z "$best" ] && return 1
  sleep "$CONFIRM_GAP"
  fb=$(free_gb "$best")
  [ -z "$fb" ] && return 1
  ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
  [ "$ok" = "1" ] || return 1
  echo "$best"
  return 0
}

wait_device() {
  local waited=0 dev
  while true; do
    dev=$(pick_confirmed_device) && { echo "$dev"; return 0; }
    if [ "$waited" -ge $(( MAX_WAIT_HOURS * 3600 / POLL )) ]; then return 1; fi
    sleep "$POLL"; waited=$((waited+1))
  done
}

launch_train() {
  local dev="$1"
  CUDA_VISIBLE_DEVICES="$dev" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision \
    --arm "$TAG" --out "$OUT" \
    --data "$DATA" --steps "$STEPS" --batch-size 4 --lr 1e-4 \
    --encoder-size 35M --seed 1 --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --head-chunk-size 0 --scorer resnet2d \
    >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$OUT.launch_pid"
  echo "$dev" > "$OUT.launch_dev"
  echo "[r2d_s1] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, confirmed twice)"
}

restarts=0
mkdir -p "$OUT"
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat "$OUT.launch_pid" 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then echo "[r2d_s1] FATAL: exceeded $MAX_RESTARTS restarts"; exit 1; fi
    dev=$(wait_device) || { echo "[r2d_s1] no device within ${MAX_WAIT_HOURS}h"; exit 1; }
    before=$(last_step)
    launch_train "$dev"
    restarts=$((restarts+1))
    # Adaptive escalation. Four attempts on 2026-09-29 all died at step ~500 on a
    # card that had cleared the bar when sampled: the node's q_fill dispatchers
    # launch into any headroom the moment it appears, so a constant threshold is
    # a coin flip. If an attempt dies without advancing the step counter - i.e.
    # it never really got going - demand more room next time instead of
    # repeating the same bet.
    sleep 300
    if [ "$(last_step)" -le "$before" ]; then
      if [ "$NEED_GB" -lt "$NEED_GB_MAX" ]; then
        NEED_GB=$((NEED_GB + 2))
        echo "[r2d_s1] died at step $(last_step) with no progress past $before; raising the admission bar to ${NEED_GB}GB"
      else
        echo "[r2d_s1] died early again but the bar is already ${NEED_GB}GB (max); continuing"
      fi
    fi
    continue
  fi
  sleep 120
done
echo "[r2d_s1] training DONE at step $STEPS $(date '+%F %T')"

EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
CKPT=$D/ckpts/${TAG}_step${STEPS}.pt
for i in $(seq 1 60); do [ -f "$CKPT" ] && break; sleep 60; done
[ -f "$CKPT" ] || { echo "[r2d_s1] FATAL: no snapshot ckpt at $CKPT"; exit 1; }

pick_gpu() {
  local best=-1 bestfree=0 c used total free
  for c in 0 1 2 3 4 5; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( total - used ))
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$c; fi
  done
  if [ "$bestfree" -ge 4096 ]; then echo "$best"; return 0; fi
  echo none; return 1
}

for split in bprna_ts0 bprna_new; do
  outd=$D/eval_decision/ow_${TAG}_step${STEPS}_${split}
  [ -f "$outd/result.json" ] && continue
  for try in $(seq 1 360); do
    gpu=$(pick_gpu) && break
    sleep 600
  done
  echo "[r2d_s1] eval $split on card $gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
    "$EPY" "$REPO/eval/ss/evaluate_decision.py" \
    --checkpoint "$CKPT" \
    --data "$D/ss_data/jsonl/${split}.jsonl" \
    --embedding-split "$split" \
    --calib-data "$D/ss_data/jsonl/bprna_vl0.jsonl" \
    --prior-weight -1 \
    --out "$outd" \
    --encoder-size 35M --device cuda --head-chunk 8 \
    --tag "ow_${TAG}_step${STEPS}_${split}"
  echo "[r2d_s1] eval $split rc=$?"
done
echo "[r2d_s1] ALL DONE $(date '+%F %T')"
