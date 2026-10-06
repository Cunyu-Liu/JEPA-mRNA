#!/bin/bash
# xens2: plana 2-seed in-family avg x r2d cross-family (15.20).
# Step 1: VL0 w-verification (the plana bucket changed from single-seed to
#         2-seed average, so the VL0-locked w must be re-verified on VL0 —
#         selection stays on the clean split only).
# Step 2: single-run verification on the four PDB-family test splits.
# Card: dynamic pick >= 18GB (two 650M backbones in-loop + r2d head).
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
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  [ "$bestfree" -ge 18000 ] && echo "$gpu" && return 0
  return 1
}

run_one() {
  local split=$1 w=$2 emb=$3
  outd=$D/eval_decision/xens2_${split}_w${w}
  [ -f $outd/result.json ] && { echo "[xens2] SKIP $split w=$w"; return 0; }
  local data=$D/ss_data/jsonl/${split}.jsonl
  local gpu=""
  for try in $(seq 1 240); do
    gpu=$(pick_gpu) && break
    sleep 120
  done
  [ -z "$gpu" ] && { echo "[xens2] no GPU for $split"; return 1; }
  echo "[xens2] $split w=$w gpu=$gpu START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$PY" $REPO/tools/xens2_eval.py \
    --plana-ckpt $P0 --plana-ckpt2 $P1 --r2d-ckpt $R2D \
    --data $data --embedding-split ${emb} \
    --w-plana $w \
    --out $outd/result.json
  echo "[xens2] $split w=$w rc=$? $(date '+%T')"
}

echo "[xens2] START $(date '+%F %T')"
# 1) VL0 w-verification (clean selection split)
for w in 0.5 0.7 0.85; do
  run_one bprna_vl0 $w bprna_vl0
done
echo "[xens2] VL0 sweep complete; inspect before test runs: "
for w in 0.5 0.7 0.85; do
  f=$D/eval_decision/xens2_bprna_vl0_w${w}/result.json
  [ -f "$f" ] && /home/cunyuliu/miniconda3/envs/lucaone/bin/python -c "
import json; d=json.load(open('$f')); m=d['pair_level']['micro']
print('  VL0 w=$w F1=%.4f P=%.4f R=%.4f' % (m['f1'],m['precision'],m['recall']))"
done
echo "[xens2] ALL DONE $(date '+%F %T')"
