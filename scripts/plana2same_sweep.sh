#!/bin/bash
# 15.23 ablation row: plana same-family 2-seed pure bucket (no r2d member).
# xens2_eval with --w-plana 1.0 => s = 0.5*plana_s0 + 0.5*plana_s1, decoded.
# Structural ablation (drop the cross-family member), not a tuned knob,
# so no VL0 selection needed — composition is fixed by the xens2 protocol.
# Runs on all 8 standard splits for the paper ablation table.
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
  [ "$bestfree" -ge 17000 ] && echo "$gpu" && return 0
  return 1
}

run_one() {
  local split=$1 emb=$2
  outd=$D/eval_decision/plana2same_${split}
  [ -f $outd/result.json ] && { echo "[p2s] SKIP $split"; return 0; }
  local gpu=""
  for try in $(seq 1 120); do
    gpu=$(pick_gpu) && break
    sleep 120
  done
  [ -z "$gpu" ] && { echo "[p2s] no GPU for $split"; return 1; }
  echo "[p2s] $split gpu=$gpu START $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 10 \
    "$PY" $REPO/tools/xens2_eval.py \
    --plana-ckpt $P0 --plana-ckpt2 $P1 --r2d-ckpt $R2D \
    --data $D/ss_data/jsonl/${split}.jsonl --embedding-split ${emb} \
    --w-plana 1.0 \
    --out $outd/result.json
  echo "[p2s] $split rc=$? $(date '+%T')"
}

echo "[p2s] START $(date '+%F %T') (plana same-family 2-seed, w=1.0)"
for split in ref_pdb_ts1 ref_pdb_ts_hard ref_pdb_ts2 ref_pdb_ts3 bprna_ts0 testsetb archiveii_embok_clean bprna_new; do
  case $split in
    archiveii_embok_clean) emb=archiveii;;
    *) emb=$split;;
  esac
  run_one $split $emb
done
echo "[p2s] ALL DONE $(date '+%F %T')"
