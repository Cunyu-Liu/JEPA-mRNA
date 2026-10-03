#!/bin/bash
# 15.12 Arm A checkpoint-selection probe: the OOD peak sits at/below 10k
# (old-corpus evidence: new@10k 0.6045 > new@20k 0.5683, and 20k-over-10k
# dropped again on the clean corpus). The launcher only auto-evaled 20k.
# This script evals the 8000-step snapshot on all six splits (watch7
# protocol, w=-1, VL0 Platt) so the peak-of-curve can be selected on VL
# evidence rather than assumed. Idempotent.
set -u
REPO=/home/cunyuliu/rna-jepa
D=/mnt/cunyuliu/rna-jepa
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
LOG=$D/runs/r2dtr1c_s8k_eval.log
exec >> "$LOG" 2>&1
echo "[s8k] START $(date '+%F %T')"

CKPT=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step8000.pt
[ -f "$CKPT" ] || { echo "[s8k] FATAL no ckpt"; exit 1; }

for split in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard; do
  outd=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s0_step8000_${split}
  [ -f $outd/result.json ] && { echo "[s8k] skip $split"; continue; }
  for try in $(seq 1 360); do
    gpu=""; bestfree=0
    for c in 0 1 2 3 4 5 7; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      free=$(( t - u ))
      if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
    done
    [ "$bestfree" -ge 8000 ] && break
    sleep 600
  done
  echo "[s8k] eval $split on card $gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
    "$PY" $REPO/eval/ss/evaluate_decision.py \
    --checkpoint $CKPT \
    --data $D/ss_data/jsonl/${split}.jsonl \
    --embedding-split ${split} \
    --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
    --prior-weight -1 \
    --out $outd \
    --encoder-size 35M --device cuda --head-chunk 8 \
    --tag ow_rinalmo_r2dtr1c_b4_s0_step8000_${split}
  echo "[s8k] eval $split rc=$?"
done
echo "[s8k] ALL DONE $(date '+%F %T')"
