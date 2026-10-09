#!/bin/bash
# T-A47b: r2d-family expansion arm — frozen-backbone resnet2d on tr1c_plus
# (tr1c 42,564 + 754 clean PDB-experimental). Byte-identical to
# launch_r2dtr1c.sh except --data/--teacher/embedding split/TAG (single
# variable: the PDB-family expansion, paired with the plana-family arm).
set -u

exec 8>/tmp/.r2d_tr1c_plus_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1c_plus.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c_plus
EMB=$D/embeddings/rinalmo-giga
TAG=rinalmo_r2dtr1cplus_b4_s0
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
NEED_GB=34
NEED_GB_MAX=38
MAX_RESTARTS=40
POLL=120
cd $REPO || exit 1

EVAL_SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb"
all_evals_done() {
  for sp in $EVAL_SPLITS; do
    [ -f $D/eval_decision/ow_${TAG}_step20000_${sp}/result.json ] || return 1
  done
  return 0
}
if all_evals_done; then
  echo "[r2d_plus] SKIP: all evals already present"; exit 0
fi

# wait for the tr1c_plus embedding shard (extractor running on GPU 7)
wait_embeddings() {
  for try in $(seq 1 240); do
    ok=1
    [ -f $EMB/bprna_tr1c_plus.shard0of1.npz ] || ok=0
    grep -q "bprna_tr1c_plus" $EMB/manifest.json 2>/dev/null || ok=0
    [ "$ok" = "1" ] && return 0
    sleep 60
  done
  return 1
}
wait_embeddings || { echo "[r2d_plus] FATAL: embeddings never appeared"; exit 1; }
[ -f $TEACHER/manifest.json ] || { echo "[r2d_plus] FATAL: no teacher manifest"; exit 1; }
mkdir -p "$OUT"

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)' 2>/dev/null || echo 0; }

pick_confirmed_device() {
  local best="" bestfb=0 c fb ok
  for c in 0 1 3 5 2 4 7 6; do
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
  echo "[r2d_plus] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, confirmed twice)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "rnajepa.train_decision.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then echo "[r2d_plus] FATAL: restarts exhausted"; exit 1; fi
    dev=$(pick_confirmed_device) || { echo "[r2d_plus] no device this cycle $(date +%T)" >&2; sleep "$POLL"; continue; }
    before=$(last_step)
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    if [ "$(last_step)" -le "$before" ]; then
      if [ "$NEED_GB" -lt "$NEED_GB_MAX" ]; then
        NEED_GB=$((NEED_GB + 2))
        echo "[r2d_plus] no progress; bar -> ${NEED_GB}GB"
      fi
    fi
    continue
  fi
  sleep 120
done
echo "[r2d_plus] training DONE at $STEPS $(date '+%F %T')"

# final eval: watch7 protocol, EIGHT splits (incl. ArchII + TestSetB), w=-1, VL0 Platt
EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
for STEP in 20000; do
  CKPT=$D/ckpts/${TAG}_step${STEP}.pt
  [ -f "$CKPT" ] || { echo "[r2d_plus] NOTE: no snapshot at $STEP"; continue; }
  for split in $EVAL_SPLITS; do
    outd=$D/eval_decision/ow_${TAG}_step${STEP}_${split}
    [ -f $outd/result.json ] && continue
    for try in $(seq 1 360); do
      gpu=""; bestfree=0
      for c in 0 1 3 5 2 4 7 6; do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
        t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
        free=$(( t - u ))
        if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
      done
      if [ "$bestfree" -ge 8000 ]; then break; fi
      sleep 600
    done
    echo "[r2d_plus] eval step$STEP $split on card $gpu $(date '+%T')"
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
      --tag ow_${TAG}_step${STEP}_${split}
    echo "[r2d_plus] eval step$STEP $split rc=$?"
  done
done
echo "[r2d_plus] ALL DONE $(date '+%F %T')"
