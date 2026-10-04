#!/bin/bash
# 15.14 cross-family ensemble sweep: plana_tr1c x r2dtr1c, score-average
# (w=0.5), exact Nussinov. Runs on every frozen split (small ones inline;
# TS0/new large -> backgrounded by the caller if needed).
# editflow env required (plana member loads multimolecule).
set -u
REPO=/home/cunyuliu/rna-jepa
D=/mnt/cunyuliu/rna-jepa
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src:$REPO/tools:$REPO/eval
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
LOG=$D/runs/xens_sweep.log
exec >> "$LOG" 2>&1
echo "[xens-sweep] START $(date '+%F %T')"

PLANA=$D/ckpts/plana_giga_tr1c_s0_step20000.pt
R2D=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt

declare -A SPLITS=(
  [ref_pdb_ts1]=ts1
  [ref_pdb_ts2]=ts2
  [ref_pdb_ts3]=ts3
  [ref_pdb_ts_hard]=ts_hard
  [testsetb]=testsetb
  [archiveii_embok_clean]=archiveii_clean
)

for SP in ref_pdb_ts3 ref_pdb_ts_hard ref_pdb_ts1 testsetb archiveii_embok_clean; do
  case $SP in
    archiveii_embok_clean) EMB=archiveii ;;
    *)                     EMB=${SP} ;;
  esac
  OUTD=$D/eval_decision/xens_${SP}
  [ -f $OUTD/result.json ] && { echo "[xens-sweep] skip $SP"; continue; }
  for try in $(seq 1 480); do
    gpu=""; bestfree=0
    for c in 0 1 2 3 4 5; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      free=$(( t - u ))
      if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
    done
    [ "$bestfree" -ge 14000 ] && break
    sleep 300
  done
  echo "[xens-sweep] $SP on gpu=$gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
    $PY $REPO/tools/xens_eval.py \
    --plana-ckpt $PLANA --r2d-ckpt $R2D \
    --data $D/ss_data/jsonl/${SP}.jsonl \
    --embedding-split ${EMB} \
    --out $OUTD/result.json
  echo "[xens-sweep] $SP rc=$?"
done

# big splits: TS0 + bpRNA-new (est. 25-45 min each)
for SP in bprna_ts0 bprna_new; do
  OUTD=$D/eval_decision/xens_${SP}
  [ -f $OUTD/result.json ] && { echo "[xens-sweep] skip $SP"; continue; }
  for try in $(seq 1 480); do
    gpu=""; bestfree=0
    for c in 0 1 2 3 4 5; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      free=$(( t - u ))
      if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
    done
    [ "$bestfree" -ge 14000 ] && break
    sleep 300
  done
  echo "[xens-sweep] $SP on gpu=$gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
    $PY $REPO/tools/xens_eval.py \
    --plana-ckpt $PLANA --r2d-ckpt $R2D \
    --data $D/ss_data/jsonl/${SP}.jsonl \
    --embedding-split ${SP} \
    --out $OUTD/result.json
  echo "[xens-sweep] $SP rc=$?"
done
echo "[xens-sweep] ALL DONE $(date '+%F %T')"
