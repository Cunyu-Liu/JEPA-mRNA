#!/bin/bash
# 15.10 Arm A: r2d_tr1 recipe on the DECONTAMINATED corpus (bprna_tr1c).
#
# Why this arm exists (measured evidence):
#   - 15.08: r2d_tr1 (frozen+2D+TR1) = bpRNA-new 0.6045, +0.104 over frozen
#     control, super-additive, UFold-level.
#   - 15.10 audit: TR1 carries exact-sequence overlap with TS0 (1087), TS1 (38),
#     TS2 (28), TS3 (18), TS-hard (21) -> the PDB-family / TS0 numbers of
#     r2d_tr1 are NOT quotable. bprna_new was clean (0 overlap), so 0.6045
#     stands.
#   - this arm re-runs the identical recipe on bprna_tr1c (exact-sequence
#     decontamination against all 9 eval splits) so that TS0 AND the PDB
#     family become quotable.
# Protocol: byte-identical to rinalmo_r2dtr1_b4_s0 (only --data/--teacher/--arm
# differ). watch7-style eval at DONE on ALL SIX target splits, w=-1, VL0 Platt.
# GPU: ~35GB peak at B=4; 33GB reservation; two-phase 34GB admission.
set -u

exec 8>/tmp/.r2d_tr1c_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1c.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c
EMB=$D/embeddings/rinalmo-giga
TAG=rinalmo_r2dtr1c_b4_s0
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
NEED_GB=34
NEED_GB_MAX=38
MAX_RESTARTS=40
POLL=120
MAX_WAIT_HOURS=96
cd $REPO || exit 1

EVAL_SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard"
all_evals_done() {
  for sp in $EVAL_SPLITS; do
    [ -f $D/eval_decision/ow_${TAG}_step20000_${sp}/result.json ] || return 1
  done
  return 0
}
if all_evals_done; then
  echo "[r2d_tr1c] SKIP: all six evals already present"; exit 0
fi

# wait for the embedding shards to exist (extractors launched 15.10)
wait_embeddings() {
  for try in $(seq 1 120); do
    ok=1
    [ -f $EMB/bprna_tr1c.shard0of2.npz ] || ok=0
    [ -f $EMB/bprna_tr1c.shard1of2.npz ] || ok=0
    # a partially-written npz fails the np.load below; only trust after
    # the extractor's manifest entry exists
    grep -q "bprna_tr1c" $EMB/manifest.json 2>/dev/null || ok=0
    [ "$ok" = "1" ] && return 0
    sleep 60
  done
  return 1
}
wait_embeddings || { echo "[r2d_tr1c] FATAL: embeddings never appeared"; exit 1; }
# teacher dir for tr1c must be complete too
[ -f $TEACHER/manifest.json ] || { echo "[r2d_tr1c] FATAL: no teacher manifest"; exit 1; }
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
  echo "[r2d_tr1c] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, confirmed twice)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "rnajepa.train_decision.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then echo "[r2d_tr1c] FATAL: restarts exhausted"; exit 1; fi
    dev=$(pick_confirmed_device) || { echo "[r2d_tr1c] no device this cycle $(date '+%T')" >&2; sleep "$POLL"; continue; }
    before=$(last_step)
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    if [ "$(last_step)" -le "$before" ]; then
      if [ "$NEED_GB" -lt "$NEED_GB_MAX" ]; then
        NEED_GB=$((NEED_GB + 2))
        echo "[r2d_tr1c] no progress; bar -> ${NEED_GB}GB"
      fi
    fi
    continue
  fi
  sleep 120
done
echo "[r2d_tr1c] training DONE at $STEPS $(date '+%F %T')"

# final eval: watch7 protocol, all six splits, w=-1, VL0 Platt
EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
for STEP in 20000; do
  CKPT=$D/ckpts/${TAG}_step${STEP}.pt
  [ -f "$CKPT" ] || { echo "[r2d_tr1c] NOTE: no snapshot at $STEP"; continue; }
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
    echo "[r2d_tr1c] eval step$STEP $split on card $gpu $(date '+%T')"
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
    echo "[r2d_tr1c] eval step$STEP $split rc=$?"
  done
done
echo "[r2d_tr1c] ALL DONE $(date '+%F %T')"
