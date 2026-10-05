#!/bin/bash
# Fill the w=0.7 protocol gap: run xens on the 4 splits missing @w=0.7
# (ts0 / bprna_new / testsetb / archiveii_embok_clean). VL0 weight sweep
# locked w=0.7 on 10-04 22:18; ts1/ts2/ts3/ts_hard already verified @0.7.
# Also sweeps VL0 at w=0.85 (VL0 is the clean selection split, allowed).
# Card 5 (21GB free). editflow env (plana member needs multimolecule).
# NOTE: plana ckpt is the tr1c variant (plana_giga_tr1c_s0), matching the
# original xens_sweep.sh protocol. archiveii_embok_clean uses EMB=archiveii.
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
PLANA=$D/ckpts/plana_giga_tr1c_s0_step20000.pt
R2D=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt

pick_gpu() {
  local gpu="" bestfree=0 c u t free
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  [ "$bestfree" -ge 14000 ] && echo "$gpu" && return 0
  return 1
}

run_one() {
  local split=$1 w=$2 emb=$3
  outd=$D/eval_decision/xens_${split}_w${w}
  [ -f $outd/result.json ] && { echo "[xensfill] SKIP $split w=$w"; return 0; }
  local data=$D/ss_data/jsonl/${split}.jsonl
  [ -f "$data" ] || { echo "[xensfill] FATAL no data $data"; return 1; }
  local gpu=""
  for try in $(seq 1 240); do
    gpu=$(pick_gpu) && break
    sleep 120
  done
  [ -z "$gpu" ] && { echo "[xensfill] no GPU for $split, giving up"; return 1; }
  echo "[xensfill] $split w=$w emb=$emb gpu=$gpu START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$PY" $REPO/tools/xens_eval.py \
    --plana-ckpt $PLANA --r2d-ckpt $R2D \
    --data $data --embedding-split ${emb} \
    --w-plana $w \
    --out $outd/result.json
  echo "[xensfill] $split w=$w rc=$? $(date '+%T')"
}

echo "[xensfill] START $(date '+%F %T')"
run_one bprna_ts0 0.7 bprna_ts0
run_one bprna_new 0.7 bprna_new
run_one testsetb 0.7 testsetb
run_one archiveii_embok_clean 0.7 archiveii
run_one bprna_vl0 0.85 bprna_vl0
echo "[xensfill] ALL DONE $(date '+%F %T')"
