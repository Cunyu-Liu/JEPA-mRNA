#!/bin/bash
# xens2 test-split verification at the VL0-confirmed w=0.7 (bucket changed
# to 2-seed plana average; VL0 re-selected 0.7: 0.8789 vs 0.8762/0.8782).
# Single run per split. Dynamic GPU pick >= 18GB.
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
W=0.7

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
  local split=$1 emb=$2
  outd=$D/eval_decision/xens2_${split}_w${W}
  [ -f $outd/result.json ] && { echo "[xens2t] SKIP $split"; return 0; }
  local data=$D/ss_data/jsonl/${split}.jsonl
  local gpu=""
  for try in $(seq 1 240); do
    gpu=$(pick_gpu) && break
    sleep 120
  done
  [ -z "$gpu" ] && { echo "[xens2t] no GPU for $split"; return 1; }
  echo "[xens2t] $split gpu=$gpu START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$PY" $REPO/tools/xens2_eval.py \
    --plana-ckpt $P0 --plana-ckpt2 $P1 --r2d-ckpt $R2D \
    --data $data --embedding-split ${emb} \
    --w-plana $W \
    --out $outd/result.json
  echo "[xens2t] $split rc=$? $(date '+%T')"
}

echo "[xens2t] START $(date '+%F %T') w=$W"
for split in ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts1 ref_pdb_ts_hard bprna_ts0 bprna_new testsetb archiveii_embok_clean; do
  case $split in
    archiveii_embok_clean) emb=archiveii;;
    *) emb=$split;;
  esac
  run_one $split $emb
done
echo "[xens2t] ALL DONE $(date '+%F %T')"
