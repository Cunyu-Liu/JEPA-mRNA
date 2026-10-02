#!/bin/bash
# r2d_tr1 (Plan-B on the 4.29x corpus): the OOD chaser arm, 15.05.
#
# Design (from measured single-axis evidence, ledger 14.49/14.57/15.00):
#   frozen RiNALMo-giga + resnet2d scorer + TR1 data (45,865 rows).
#   - frozen backbone keeps family generality (14.77: adaptation costs
#     -0.071 OOD; the frozen arm holds 0.501);
#   - 2D scorer: +0.067 both splits (14.74);
#   - TR1 data: +0.029 OOD at 20k, and the ff-family 40k point showed the
#     OOD peak is at/below 20k steps for this corpus size (14.56: 20k
#     0.5162 -> 40k 0.4999) -> 20,000 steps is the target, snapshots at
#     10k/15k give the early side of the curve;
#   - the 14.57 negative interaction (bigtr1 0.4558) was CAPACITY x data
#     on the flat head; this arm keeps the small (811k) scorer, avoiding
#     that axis entirely.
# Protocol: byte-identical to rinalmo_r2d_b4_s0 (only --data/--arm/seed
# dir differ); watch7 eval protocol at DONE, both splits, w=-1.
# GPU: needs ~35GB peak at B=4 (measured r2d family); 33GB reservation.
set -u

exec 8>/tmp/.r2d_tr1_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1
EMB=$D/embeddings/rinalmo-giga
TAG=rinalmo_r2dtr1_b4_s0
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
NEED_GB=34
NEED_GB_MAX=38
MAX_RESTARTS=40
POLL=120
MAX_WAIT_HOURS=96
cd $REPO || exit 1

if [ -f $D/eval_decision/ow_${TAG}_step20000_bprna_ts0/result.json ] && \
   [ -f $D/eval_decision/ow_${TAG}_step20000_bprna_new/result.json ]; then
  echo "[r2d_tr1] SKIP: both evals already present"; exit 0
fi
mkdir -p "$OUT"

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0; }

pick_confirmed_device() {
  local best="" bestfb=0 c fb ok
  for c in 0 1 3 5 2 4; do
    fb=$(free_gb "$c"); [ -z "$fb" ] && continue
    ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
    if [ "$ok" = "1" ]; then
      if awk -v a="$fb" -v b="$bestfb" 'BEGIN{exit !(a>b)}'; then bestfb=$fb; best=$c; fi
    fi
  done
  [ -z "$best" ] && return 1
  sleep 90
  fb=$(free_gb "$best")
  [ -z "$fb" ] && return 1
  ok=$(awk -v x="$fb" -v n="$NEED_GB" 'BEGIN{print (x>=n)?1:0}')
  [ "$ok" = "1" ] || return 1
  echo "$best"; return 0
}

launch_train() {
  local dev="$1"
  CUDA_VISIBLE_DEVICES="$dev" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision \
    --arm "$TAG" --out "$OUT" \
    --data "$DATA" --steps "$STEPS" --batch-size 4 --lr 1e-4 \
    --encoder-size 35M --seed 0 --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --head-chunk-size 0 --scorer resnet2d --gpu-reserve-gb 33 \
    9>&- >> "$LOG" 2>&1 < /dev/null &
  echo $! > $OUT.launch_pid
  echo "$dev" > $OUT.launch_dev
  echo "[r2d_tr1] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, confirmed twice)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "rnajepa.train_decision.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then echo "[r2d_tr1] FATAL: restarts exhausted"; exit 1; fi
    dev=$(pick_confirmed_device) || { echo "[r2d_tr1] no device this cycle $(date '+%T')" >&2; sleep "$POLL"; continue; }
    before=$(last_step)
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    if [ "$(last_step)" -le "$before" ]; then
      if [ "$NEED_GB" -lt "$NEED_GB_MAX" ]; then
        NEED_GB=$((NEED_GB + 2))
        echo "[r2d_tr1] no progress; bar -> ${NEED_GB}GB"
      fi
    fi
    continue
  fi
  sleep 120
done
echo "[r2d_tr1] training DONE at $STEPS $(date '+%F %T')"

# final eval: watch7 protocol, both splits, w=-1, VL0 Platt
EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
for STEP in 10000 20000; do
  CKPT=$D/ckpts/${TAG}_step${STEP}.pt
  [ -f "$CKPT" ] || { echo "[r2d_tr1] NOTE: no snapshot at $STEP"; continue; }
  for split in bprna_ts0 bprna_new; do
    outd=$D/eval_decision/ow_${TAG}_step${STEP}_${split}
    [ -f $outd/result.json ] && continue
    for try in $(seq 1 360); do
      gpu=""; bestfree=0
      for c in 0 1 3 5 2 4; do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
        t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
        free=$(( t - u ))
        if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
      done
      if [ "$bestfree" -ge 8000 ]; then break; fi
      sleep 600
    done
    echo "[r2d_tr1] eval step$STEP $split on card $gpu $(date '+%T')"
    env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
      "$EPY" $REPO/eval/ss/evaluate_decision.py \
      --checkpoint $CKPT \
      --data $D/ss_data/jsonl/${split}.jsonl \
      --embedding-split $split \
      --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
      --prior-weight -1 \
      --out $outd \
      --encoder-size 35M --device cuda --head-chunk 8 \
      --tag ow_${TAG}_step${STEP}_${split}
    echo "[r2d_tr1] eval step$STEP $split rc=$?"
  done
done
echo "[r2d_tr1] ALL DONE $(date '+%F %T')"
