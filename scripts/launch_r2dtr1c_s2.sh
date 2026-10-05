#!/bin/bash
# r2dtr1c seed=2: same recipe as s0, different seed for basin diversity.
# s1 (seed=1) landed in a worse basin (TS2 0.6277 vs s0 0.7686); s2 is a fresh
# draw. Polls cards 0-5 for 34GB CUDA-free (skip 6/7 MIG). Daemon-style, one-shot.
set -u

exec 8>/tmp/.r2dtr1c_s2_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1c.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c
EMB=$D/embeddings/rinalmo-giga
TAG=rinalmo_r2dtr1c_b4_s2
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
NEED_GB=34
POLL=120
MAX_WAIT_HOURS=120
cd $REPO || exit 1

mkdir -p "$OUT" "$D/ckpts" "$D/eval_decision"

# check if already done
EVAL_SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard"
all_evals_done() {
  for sp in $EVAL_SPLITS; do
    [ -f $D/eval_decision/ow_${TAG}_step${STEPS}_${sp}/result.json ] || return 1
  done
  return 0
}
if all_evals_done; then
  echo "[s2] SKIP: all six evals already present"; exit 0
fi

# verify embeddings + teacher
[ -f $EMB/bprna_tr1c.shard0of2.npz ] || { echo "[s2] FATAL: no emb shard0"; exit 1; }
[ -f $EMB/bprna_tr1c.shard1of2.npz ] || { echo "[s2] FATAL: no emb shard1"; exit 1; }
[ -f $TEACHER/manifest.json ] || { echo "[s2] FATAL: no teacher manifest"; exit 1; }

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | "$PY" -c "
import json,sys
try: print(json.loads(sys.stdin.read()).get('step',0))
except Exception: print(0)" 2>/dev/null || echo 0; }

pick_confirmed_device() {
  local best="" bestfb=0 c fb ok
  for c in 0 1 3 5 2 4; do  # skip 6/7 (MIG-sliced)
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
    --encoder-size 35M --seed 2 --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --head-chunk-size 0 --scorer resnet2d --gpu-reserve-gb 33 \
    --grad-clip 1.0 \
    9>&- >> "$LOG" 2>&1 < /dev/null &
  echo $! > $OUT.launch_pid
  echo "[s2] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "rnajepa.train_decision.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    dev=$(pick_confirmed_device) || { echo "[s2] no device this cycle $(date '+%T')" >&2; sleep "$POLL"; continue; }
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    continue
  fi
  sleep 120
done
echo "[s2] TRAIN DONE at $STEPS $(date '+%F %T')"

# snapshot
CKPT=$D/ckpts/${TAG}_step${STEPS}.pt
[ -f "$OUT/resume.pt" ] && [ ! -f "$CKPT" ] && ln -sf "$OUT/resume.pt" "$CKPT"
[ -f "$CKPT" ] || { echo "[s2] FATAL: no resume.pt"; exit 1; }

# 6-split eval (use whatever card has >=8GB CUDA-free, skip 6/7)
EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
for split in $EVAL_SPLITS; do
  outd=$D/eval_decision/ow_${TAG}_step${STEPS}_${split}
  [ -f $outd/result.json ] && continue
  # find a card with >=8GB CUDA-free (skip 6/7)
  gpu=""; bestfree=0
  for c in 0 1 3 5 2 4; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  [ -z "$gpu" ] && { echo "[s2] no GPU for eval $split"; continue; }
  echo "[s2] eval $split on card $gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
    "$EPY" $REPO/eval/ss/evaluate_decision.py \
    --checkpoint $CKPT \
    --data $D/ss_data/jsonl/${split}.jsonl \
    --embedding-split ${split} \
    --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
    --prior-weight -1 \
    --out $outd \
    --encoder-size 35M --device cuda --head-chunk 8 \
    --tag ow_${TAG}_step${STEPS}_${split}
  echo "[s2] eval $split rc=$? $(date '+%T')"
done
echo "[s2] ALL DONE $(date '+%F %T')"
