#!/bin/bash
# 15.11 Tier-1 chaser: ArchiveII + TS0 ensemble + Arm-A ArchiveII top-up.
#
# Facts established before writing this (all on disk):
#   - archiveii_embok_clean (2,214 seqs) has ZERO overlap with bprna_tr0 AND
#     bprna_tr1c -> both Plan-A family and Arm A (r2dtr1c) are quotable on it.
#   - UFold/MXfold2/ETeRNAfold are NOT yet measured on the clean ArchiveII
#     buckets -> tier-1 SOTA reference row missing; this script fills it via
#     the existing run_ufold adapter (same project scorer).
#   - ext40k curve is DECLINING on TS0 (20k 0.7268 -> 30k 0.7210 -> 40k 0.7178):
#     more training is not the TS0 lever; ensemble + better data are.
#   - ensemble_eval.py averages raw score matrices over K same-config ckpts
#     then decodes once (exact Nussinov). Only used for frozen-head arms.
#   - plana arms are in-loop-backbone models: ensemble_eval does NOT apply;
#     plana ensemble would need a new tool (deferred, not in this script).
#   - r2dtr1c (Arm A) evals archiveii_embok_clean with the watch7 protocol
#     AFTER its 6-split eval -- appending, not editing, the running launcher.
set -u
REPO=/home/cunyuliu/rna-jepa
D=/mnt/cunyuliu/rna-jepa
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/mnt/cunyuliu/pylibs:$REPO/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2
LOG=$D/runs/tier1_arch.log
mkdir -p $D/runs
exec >> "$LOG" 2>&1
echo "[tier1] START $(date '+%F %T')"

wait_gpu () {
  local need_gb=$1 gpu bestfree
  for try in $(seq 1 480); do
    gpu=""; bestfree=0
    for c in 0 1 3 5 2 4 7 6; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
      free=$(( t - u ))
      if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
    done
    [ "$bestfree" -ge "$need_gb" ] && { echo "$gpu"; return 0; }
    sleep 300
  done
  return 1
}

run_on_gpu () {
  local need_gb=$1; shift
  local GPU
  GPU=$(wait_gpu "$need_gb") || { echo "[tier1] NO GPU, skip: $*"; return 1; }
  echo "[tier1] gpu=$GPU :: $*"
  env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR OMP_NUM_THREADS=2 nice -n 15 "$@"
  return $?
}

# ---------- 1) UFold + MXfold2 on clean ArchiveII (tier-1 SOTA reference) ----------
for B in le100 gt100_le200 gt200_le400 gt400; do
  OUT=$D/eval_decision/baselines_ufold_archiveii_embok_clean_${B}.json
  [ -f "$OUT" ] && { echo "[tier1] skip ufold $B"; continue; }
  run_on_gpu 8000 $PY $REPO/eval/ss/run_ufold.py --split archiveii_embok_clean_${B} --eval-mode
  echo "[tier1] ufold $B rc=$?"
done

# ---------- 2) r2d family (frozen head, s0+s1) ensemble on TS0 + new + archiveii ----------
# embedding-split is "archiveii" (the full-set store; embok_clean is its subset
# and EmbeddingStore looks up by sequence) — same convention as
# run_clean_split_evals.sh.
CK0=$D/ckpts/rinalmo_r2dtr1_b4_s0_step10000.pt
CK1=$D/ckpts/rinalmo_r2dtr1_b4_s1_step10000.pt
for SPL in bprna_ts0 bprna_new archiveii_embok_clean; do
  OUTD=$D/eval_decision/ens_r2dtr1_2seed_${SPL}
  [ -f $OUTD/result.json ] && { echo "[tier1] skip ens $SPL"; continue; }
  case $SPL in
    archiveii_embok_clean) EMB=archiveii ;;
    *)                     EMB=${SPL} ;;
  esac
  run_on_gpu 10000 $PY $REPO/tools/ensemble_eval.py \
    --checkpoints $CK0 $CK1 \
    --data $D/ss_data/jsonl/${SPL}.jsonl \
    --split ${EMB} \
    --out $OUTD
  echo "[tier1] ens $SPL rc=$?"
done

echo "[tier1] ALL DONE $(date '+%F %T')"
