#!/bin/bash
# VL0 sweep for the plana-bucket inner weight (s1w): 0.3 / 0.5 / 0.7.
# 15.20 ran s1w=0.5 (plain average); TS3 dipped (s1 weak on TS3) —
# asymmetric weights may recover it. Selection split only (VL0).
# Avoids GPU1 (plana_s2 training). Card pick >= 18GB among 0/2/3/4/5.
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
P0=$D/ckpts/plana_giga_tr1c_s0_step20000.pt
P1=$D/ckpts/plana_giga_tr1c_s1_step20000.pt
R2D=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt

pick_gpu() {
  local gpu="" bestfree=0 c u t free
  for c in 0 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  [ "$bestfree" -ge 18000 ] && echo "$gpu" && return 0
  return 1
}

run_one() {
  local s1w=$1
  outd=$D/eval_decision/xens2_vl0_s1w${s1w}
  [ -f $outd/result.json ] && { echo "[s1wsweep] SKIP s1w=$s1w"; return 0; }
  local gpu=""
  for try in $(seq 1 240); do
    gpu=$(pick_gpu) && break
    sleep 120
  done
  [ -z "$gpu" ] && { echo "[s1wsweep] no GPU"; return 1; }
  echo "[s1wsweep] s1w=$s1w gpu=$gpu START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$PY" $REPO/tools/xens2_eval.py \
    --plana-ckpt $P0 --plana-ckpt2 $P1 --r2d-ckpt $R2D \
    --data $D/ss_data/jsonl/bprna_vl0.jsonl --embedding-split bprna_vl0 \
    --w-plana 0.7 --plana-s1-weight $s1w \
    --out $outd/result.json
  echo "[s1wsweep] s1w=$s1w rc=$? $(date '+%T')"
}

echo "[s1wsweep] START $(date '+%F %T')"
for w in 0.3 0.5 0.7; do run_one $w; done
echo "[s1wsweep] ALL DONE $(date '+%F %T')"
