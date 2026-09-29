#!/bin/bash
# Plan-B second seed: rinalmo_r2d_b4_s1 (resnet2d scorer, frozen RiNALMo-giga,
# identical protocol to rinalmo_r2d_b4_s0 / launch_plan_b.sh; ONLY seed differs).
# Wait-for-device daemon: the 2D-scorer head path needs ~35GB at B=4 (14.76),
# so only a nearly-empty full card qualifies. Keep-alive watcher relaunches on
# silent death (train_decision auto-resumes from resume.pt). On DONE: auto-eval
# TS0 + bpRNA-new with the watch7 protocol (w=-1, VL0 Platt, exact decode).
set -u

# single-instance guard: duplicate invocations must exit
exec 9>/mnt/cunyuliu/rna-jepa/runs/.launch_r2d_s1.lock
flock -n 9 || exit 0
PY=/home/cunyuliu/miniconda3/envs/env_placeholder/bin/python
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
NEED_GB=31
POLL=120
MAX_WAIT_HOURS=120
MAX_RESTARTS=8
cd $REPO || exit 1

# idempotency: exit only when there is truly nothing left to do (both evals
# done). A live trainer must NOT cause an exit - the daemon's job is to watch
# it (keep-alive + final eval); the watch loop adopts it via the pid file.
if [ -f $D/eval_decision/ow_${TAG}_step20000_bprna_ts0/result.json ] && [ -f $D/eval_decision/ow_${TAG}_step20000_bprna_new/result.json ]; then echo "SKIP: already evaluated"; exit 0; fi
if ! pgrep -f "rnajepa.train_decision.*--out $OUT" > /dev/null && [ ! -f $OUT/resume.pt ]; then
  :  # fresh start - the watch loop will relaunch from scratch
fi

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0; }

wait_device() {
  local waited=0
  while true; do
    for c in 0 1 2 3 4 5; do
      local fb; fb=$(free_gb "$c")
      if [ -n "$fb" ]; then
        local ok; ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
        if [ "$ok" = "1" ]; then echo "$c"; return 0; fi
      fi
    done
    if [ "$waited" -ge $(( MAX_WAIT_HOURS * 3600 / POLL )) ]; then return 1; fi
    sleep $POLL; waited=$((waited+1))
  done
}

launch_train() {
  local dev=$1
  CUDA_VISIBLE_DEVICES=$dev PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision     --arm $TAG --out $OUT     --data $DATA --steps $STEPS --batch-size 4 --lr 1e-4     --encoder-size 35M --seed 1 --device cuda     --log-every 25 --save-every 500     --teacher-dir "$TEACHER"     --embedding-dir "$EMB" --embedding-d-model 1280     --head-chunk-size 0 --scorer resnet2d     >> "$LOG" 2>&1 < /dev/null &
  echo $! > $OUT.launch_pid
  echo "$dev" > $OUT.launch_dev
  echo "[r2d_s1] launched pid=$! on $dev $(date '+%F %T')"
}

restarts=0
mkdir -p $OUT
while [ "$(last_step)" -lt $STEPS ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if [ "$restarts" -ge $MAX_RESTARTS ]; then echo "[r2d_s1] FATAL: exceeded restarts"; exit 1; fi
    dev=$(wait_device) || { echo "[r2d_s1] no device within ${MAX_WAIT_HOURS}h"; exit 1; }
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 60
    continue
  fi
  sleep 120
done
echo "[r2d_s1] training DONE at step $STEPS $(date '+%F %T')"

# final eval (watch7 protocol), snapshot ckpt appears via monitor cron (<=10 min)
EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/rna-jepa/tmp 2>/dev/null || export TMPDIR=/tmp
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
CKPT=$D/ckpts/${TAG}_step${STEPS}.pt
for i in $(seq 1 60); do [ -f $CKPT ] && break; sleep 60; done
[ -f $CKPT ] || { echo "[r2d_s1] FATAL: no snapshot ckpt at $CKPT"; exit 1; }

pick_gpu() {
  local best=-1 bestfree=0
  for c in 0 1 2 3 4 5; do
    local used total free
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
  [ -f $outd/result.json ] && continue
  for try in $(seq 1 360); do
    gpu=$(pick_gpu) && break
    sleep 600
  done
  echo "[r2d_s1] eval $split on card $gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15     "$EPY" $REPO/eval/ss/evaluate_decision.py     --checkpoint $CKPT     --data $D/ss_data/jsonl/${split}.jsonl     --embedding-split $split     --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl     --prior-weight -1     --out $outd     --encoder-size 35M --device cuda --head-chunk 8     --tag ow_${TAG}_step${STEPS}_${split}
  echo "[r2d_s1] eval $split rc=$?"
done
echo "[r2d_s1] ALL DONE $(date '+%F %T')"
