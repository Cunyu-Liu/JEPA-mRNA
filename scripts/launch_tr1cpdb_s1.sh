#!/bin/bash
# T-A28: PDB-family data-fortified arm — r2d recipe (frozen giga + resnet2d)
# on bprna_tr1c_pdb (tr1c + 234 clean pdb669 rows). Single variable vs
# rinalmo_r2dtr1c_b4_s0: training-corpus composition (+0.55% PDB rows).
# Evidence base: 15.26 analysis (TS2/TS3 gap = PDB-family stems our head has
# never been trained to fire on; RNAformer wins via family-specialised data).
# Auto 6-split eval at DONE (w=-1, VL0 Platt). Arm tag: rinalmo_r2dtr1cpdb_b4_s0.
set -u

SUFFIX=tr1cpdb_s1
LAMBDA_FLAG=""
TAG=rinalmo_r2dtr1c_${SUFFIX}_b4_s0
exec 8>/tmp/.r2dtr1c_${SUFFIX}_daemon.lock
flock -n 8 || exit 0

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/home/cunyuliu/rna-jepa/src
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr1c_pdb.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c_pdb
EMB=$D/embeddings/rinalmo-giga
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
NEED_GB=34
POLL=120
cd $REPO || exit 1

mkdir -p "$OUT" "$D/ckpts" "$D/eval_decision"

EVAL_SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard"
all_evals_done() {
  for sp in $EVAL_SPLITS; do
    [ -f $D/eval_decision/ow_${TAG}_step${STEPS}_${sp}/result.json ] || return 1
  done
  return 0
}
if all_evals_done; then
  echo "[${SUFFIX}] SKIP: all six evals already present"; exit 0
fi

[ -f $EMB/bprna_tr1c_pdb.shard0of1.npz ] || { echo "[${SUFFIX}] FATAL: no emb shard"; exit 1; }
[ -f $TEACHER/manifest.json ] || { echo "[${SUFFIX}] FATAL: no teacher manifest"; exit 1; }

free_gb() { CUDA_VISIBLE_DEVICES="$1" "$PY" -c "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null; }
last_step() { tail -1 $OUT/train_log.jsonl 2>/dev/null | "$PY" -c '
import json,sys
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
    --encoder-size 35M --seed 1 --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --head-chunk-size 0 --scorer resnet2d --gpu-reserve-gb 33 \
    --grad-clip 1.0 \
    9>&- >> "$LOG" 2>&1 < /dev/null &
  echo $! > $OUT.launch_pid
  echo "[${SUFFIX}] launched pid=$! on $dev $(date '+%F %T') (bar=${NEED_GB}GB, corpus=tr1c+pdb669_clean)"
}

restarts=0
while [ "$(last_step)" -lt "$STEPS" ]; do
  pid=$(cat $OUT.launch_pid 2>/dev/null || echo '')
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    if pgrep -f "rnajepa.train_decision.*--out $OUT" > /dev/null; then
      sleep 120; continue
    fi
    if [ "$restarts" -ge 40 ]; then echo "[${SUFFIX}] FATAL: restarts exhausted (s1)"; exit 1; fi
    dev=$(pick_confirmed_device) || { echo "[${SUFFIX}] no device this cycle $(date '+%T')" >&2; sleep "$POLL"; continue; }
    launch_train "$dev"
    restarts=$((restarts+1))
    sleep 300
    continue
  fi
  sleep 120
done
echo "[${SUFFIX}] TRAIN DONE at $STEPS $(date '+%F %T')"

CKPT=$D/ckpts/${TAG}_step${STEPS}.pt
[ -f "$OUT/resume.pt" ] && [ ! -f "$CKPT" ] && ln -sf "$OUT/resume.pt" "$CKPT"
[ -f "$CKPT" ] || { echo "[${SUFFIX}] FATAL: no resume.pt"; exit 1; }

EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
for split in $EVAL_SPLITS; do
  outd=$D/eval_decision/ow_${TAG}_step${STEPS}_${split}
  [ -f $outd/result.json ] && continue
  gpu=""; bestfree=0
  for c in 0 1 3 5 2 4; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  [ -z "$gpu" ] && { echo "[${SUFFIX}] no GPU for eval $split"; continue; }
  echo "[${SUFFIX}] eval $split on card $gpu $(date '+%T')"
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
  echo "[${SUFFIX}] eval $split rc=$? $(date '+%T')"
done
echo "[${SUFFIX}] ALL DONE $(date '+%F %T')"
