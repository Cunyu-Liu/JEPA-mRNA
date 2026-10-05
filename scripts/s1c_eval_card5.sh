#!/bin/bash
# s1c 6-split eval locked to card 5 (card 6/7 are MIG-sliced, CUDA sees 1.2GB)
# One-shot, no daemon loop, no flock.
set -u
EPY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
TAG=rinalmo_r2dtr1c_b4_s1
CKPT=$D/ckpts/${TAG}_step20000.pt

EVAL_SPLITS="bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard"
mkdir -p $D/eval_decision
echo "[s1c_eval] START $(date '+%F %T') card=5 ckpt=$CKPT"
for split in $EVAL_SPLITS; do
  outd=$D/eval_decision/ow_${TAG}_step20000_${split}
  if [ -f $outd/result.json ]; then
    echo "[s1c_eval] SKIP $split (result.json exists)"; continue
  fi
  rm -f $outd/result.json.tmp 2>/dev/null
  echo "[s1c_eval] $split START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$EPY" $REPO/eval/ss/evaluate_decision.py \
    --checkpoint $CKPT \
    --data $D/ss_data/jsonl/${split}.jsonl \
    --embedding-split ${split} \
    --calib-data $D/ss_data/jsonl/bprna_vl0.jsonl \
    --prior-weight -1 \
    --out $outd \
    --encoder-size 35M --device cuda --head-chunk 8 \
    --tag ow_${TAG}_step20000_${split}
  rc=$?
  echo "[s1c_eval] $split rc=$rc $(date '+%T')"
  if [ $rc -ne 0 ]; then
    echo "[s1c_eval] WARN $split rc=$rc, continuing to next split"
  fi
done
echo "[s1c_eval] ALL DONE $(date '+%F %T')"
