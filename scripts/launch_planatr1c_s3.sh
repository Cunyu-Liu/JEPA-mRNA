#!/bin/bash
# 15.11 Arm B: Plan-A (gradual unfreeze giga backbone) on the DECONTAMINATED
# corpus bprna_tr1c — the TS0 strike arm.
#
# Why (measured evidence):
#   - Plan-A is the TS0 specialist family: 0.7268-0.7336 single-seed on TS0
#     vs r2d family's 0.66-0.68 — the adapted backbone buys ~+0.06 TS0.
#   - Plan-A trains on tr0 only (10,682 rows). tr1c gives 4x clean data.
#   - 14.57 taught: capacity x data interacts NEGATIVELY on the flat head.
#     Here the data axis is applied to the ADAPTED-BACKBONE arm whose head
#     is small (resnet2d, 811k) — the axis pair (adaptation x data) has
#     never been measured; this arm measures it with everything else held
#     byte-identical to plana_giga_s0 (only --data/--teacher differ).
#   - TS0 target: RNAformer 0.7578 project-GT. Plan-A tr0 3-seed mean
#     0.7283±0.0047 -> this arm needs ~+0.03 from 4x clean data.
# Protocol: identical to launch_plan_a.sh (warmup 1600, unfreeze every 800
# by 2, head-lr 1e-4, backbone-lr 1e-5, 20k steps, B=4). eval via
# eval_plan_a.py (in-loop backbone; splits ts0,new + PDB family) after DONE.
set -u

exec 8>/tmp/.plana_tr1c_s3_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1c.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c
OUT=$D/runs/plana_giga_tr1c_s3
LOG=$D/runs/plana_giga_tr1c_s3.log
STEPS=20000
NEED_GB=16
cd $REPO || exit 1

if [ -f $D/eval_decision/plana_giga_tr1c_s3_step20000/result.json ]; then
  echo "[plana_tr1c_s3] SKIP: eval already present"; exit 0
fi
mkdir -p "$OUT"

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0; }

# full cards only (0-5); MIG slices 6/7 cannot host the 650M backbone
pick_confirmed_device() {
  local best="" bestfb=0 c fb ok
  for c in 0 1 2 3 4 5; do
    fb=$(free_gb "$c"); [ -z "$fb" ] && continue
    ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
    if [ "$ok" = "1" ]; then
      if awk -v a="$fb" -v b="$bestfb" 'BEGIN{exit !(a>b)}'; then bestfb=$fb; best=$c; fi
    fi
  done
  [ -z "$best" ] && return 1
  sleep 90
  fb=$(free_gb "$best")
  ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
  [ "$ok" = "1" ] || return 1
  echo "$best"; return 0
}

launch_train() {
  local dev="$1"
  CUDA_VISIBLE_DEVICES="$dev" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup nice -n 5 "$PY" tools/train_plan_a.py \
    --out "$OUT" \
    --data "$DATA" \
    --teacher-dir "$TEACHER" \
    --steps "$STEPS" \
    --batch-size 4 \
    --head-lr 1e-4 --backbone-lr 1e-5 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 --seed 3 \
    --save-every 500 --snapshot-every 2000 \
    9>&- >> "$LOG" 2>&1 < /dev/null &
  echo $! > $OUT.launch_pid
  echo "$dev" > $OUT.launch_dev
  echo "[plana_tr1c_s3] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, confirmed twice)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "train_plan_a.py.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    if [ "$restarts" -ge 40 ]; then echo "[plana_tr1c_s3] FATAL: restarts exhausted"; exit 1; fi
    dev=$(pick_confirmed_device) || { echo "[plana_tr1c_s3] no device this cycle $(date '+%T')" >&2; sleep 300; continue; }
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    continue
  fi
  sleep 120
done
echo "[plana_tr1c_s3] training DONE at $STEPS $(date '+%F %T')"

# final eval: eval_plan_a (in-loop backbone protocol), tier-1 + PDB family
EPY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src:$REPO/tools
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
CKPT=$D/ckpts/plana_giga_tr1c_s3_step${STEPS}.pt
OUTD=$D/eval_decision/plana_giga_tr1c_s3_step${STEPS}
mkdir -p $OUTD
for try in $(seq 1 360); do
  gpu=""; bestfree=0
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
  done
  [ "$bestfree" -ge 16000 ] && break
  sleep 600
done
echo "[plana_tr1c_s3] eval on card $gpu $(date '+%T')"
env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
  "$EPY" $REPO/tools/eval_plan_a.py \
  --checkpoint $CKPT \
  --splits ts0,new,ts1,hard,ts2,ts3 \
  --dump-per-seq \
  --out $OUTD/plan_a_result.json
echo "[plana_tr1c_s3] eval rc=$?"
echo "[plana_tr1c_s3] ALL DONE $(date '+%F %T')"
