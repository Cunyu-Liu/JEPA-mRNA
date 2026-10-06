#!/bin/bash
# Robust re-run of the plana_tr1c_s1 6-split eval (the daemon's attempt
# OOM'd on card 0 when an external process filled it mid-run).
# Retry-per-split with card re-pick on failure; skip when result exists.
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
D=/mnt/cunyuliu/rna-jepa
REPO=/home/cunyuliu/rna-jepa
CKPT=$D/ckpts/plana_giga_tr1c_s1_step20000.pt
OUTD=$D/eval_decision/plana_giga_tr1c_s1_step20000
mkdir -p $OUTD

pick_gpu() {
  local gpu="" bestfree=0 c u t free
  for c in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
    free=$(( t - u ))
    [ "$free" -gt "$bestfree" ] && { bestfree=$free; gpu=$c; }
  done
  [ "$bestfree" -ge 16000 ] && echo "$gpu" && return 0
  return 1
}

echo "[s1eval] START $(date '+%F %T')"
if [ -f $OUTD/plan_a_result.json ]; then
  echo "[s1eval] SKIP: plan_a_result.json already present"; exit 0
fi
for attempt in $(seq 1 20); do
  gpu=""
  for try in $(seq 1 120); do
    gpu=$(pick_gpu) && break
    sleep 120
  done
  [ -z "$gpu" ] && { echo "[s1eval] no GPU after 20 tries"; exit 1; }
  echo "[s1eval] attempt $attempt on card $gpu $(date '+%T')"
  env CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 \
    "$PY" $REPO/tools/eval_plan_a.py \
    --checkpoint $CKPT \
    --splits ts0,new,ts1,hard,ts2,ts3 \
    --dump-per-seq \
    --out $OUTD/plan_a_result.json
  rc=$?
  echo "[s1eval] attempt $attempt rc=$rc $(date '+%T')"
  [ $rc -eq 0 ] && [ -f $OUTD/plan_a_result.json ] && { echo "[s1eval] SUCCESS"; exit 0; }
  sleep 300
done
echo "[s1eval] FAILED after 20 attempts"
exit 1
