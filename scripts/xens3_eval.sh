#!/bin/bash
# 3-way cross-family ensemble: plana_tr1c x (0.5*r2dtr1c_s0 + 0.5*r2dtr1c_s1)
# Runs on VL0 (sanity, w-plana=0.7 locked from 2-way sweep) + TS2 + TS3
# (the two remaining SOTA gaps). editflow env for plana multimolecule.
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
PLANA=$D/ckpts/plana_giga_s0_step20000.pt
R2D_S0=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt
R2D_S1=$D/ckpts/rinalmo_r2dtr1c_b4_s1_step20000.pt
W=0.7
GPU=5

# VL0 sanity first (clean, zero-overlap with all test splits)
SPLITS="bprna_vl0 ref_pdb_ts2 ref_pdb_ts3"
mkdir -p $D/eval_decision
echo "[xens3] START $(date '+%F %T') card=$GPU w_plana=$W"
for split in $SPLITS; do
  outd=$D/eval_decision/xens3_${split}
  if [ -f $outd/result.json ]; then
    echo "[xens3] SKIP $split (result.json exists)"; continue
  fi
  echo "[xens3] $split START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$PY" $REPO/tools/xens_eval.py \
    --plana-ckpt $PLANA --r2d-ckpt $R2D_S0 --r2d-ckpt2 $R2D_S1 \
    --data $D/ss_data/jsonl/${split}.jsonl \
    --embedding-split ${split} \
    --w-plana $W \
    --out $outd/result.json
  rc=$?
  echo "[xens3] $split rc=$rc $(date '+%T')"
  if [ $rc -ne 0 ]; then
    echo "[xens3] ERROR $split failed, continuing"
    continue
  fi
  $PY -c "import json; d=json.load(open('$outd/result.json')); m=d['pair_level']['micro']; print('  $split F1=%.4f P=%.4f R=%.4f' % (m['f1'],m['precision'],m['recall']))" 2>/dev/null
done
echo "[xens3] ALL DONE $(date '+%F %T')"
