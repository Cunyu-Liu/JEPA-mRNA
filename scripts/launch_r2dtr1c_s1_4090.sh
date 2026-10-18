#!/bin/bash
# New-cluster (RTX 4090 24GB) retrain of rinalmo_r2dtr1c_b4_s1 (seed 1).
# The A100 arm needed ~35GB peak at B=4 (resnet2d no column chunking); on 24GB
# we lower to B=2 with grad accumulation off (recipe fidelity: same optimizer,
# lr, steps, scorer, data, teacher, embeddings; only batch fits the card).
# NOTE: batch-size is a recipe change vs the original arm -> log it explicitly.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/root/rna_jev/src
D=/root/rna_jepa_data
REPO=/root/rna_jev
DATA=$D/ss_data/jsonl/bprna_tr1c.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr1c
EMB=$D/embeddings/rinalmo-giga
TAG=rinalmo_r2dtr1c_b4_s1
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log
STEPS=20000
cd $REPO || exit 1
mkdir -p "$OUT"
exec 8>/tmp/.r2dtr1c_s1_4090.lock
flock -n 8 || { echo "[s1_4090] another instance holds the lock"; exit 0; }

if [ -f $OUT/train_log.jsonl ] && [ "$(tail -1 $OUT/train_log.jsonl 2>/dev/null | $PY -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step",0))
except Exception: print(0)')" -ge "$STEPS" ]; then
  echo "[s1_4090] already at $STEPS steps"; exit 0
fi

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision \
  --arm "$TAG" --out "$OUT" \
  --data "$DATA" --steps "$STEPS" --batch-size 2 --lr 1e-4 \
  --encoder-size 35M --seed 1 --device cuda \
  --log-every 25 --save-every 500 \
  --teacher-dir "$TEACHER" \
  --embedding-dir "$EMB" --embedding-d-model 1280 \
  --head-chunk-size 0 --scorer resnet2d --gpu-reserve-gb 20 \
  9>&- >> "$LOG" 2>&1 < /dev/null &
echo $! > $OUT.launch_pid
echo "[s1_4090] launched pid=$! on GPU0 $(date '+%F %T') batch=2 (4090 24GB adaptation, seed=1)"
