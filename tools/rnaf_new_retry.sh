#!/bin/bash
# RNAformer on bprna_new: retry loop until the whole split is evaluated.
# The pass is resumable only from scratch (adapter has no partial state),
# but each attempt is ~15 min; contention on this node is intermittent, so
# retry-on-any-card until the dbn lands. 40 attempts, 2-min gaps.
set -u
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
M=/mnt/cunyuliu/rna-jepa/refmodels/models
O=/mnt/cunyuliu/rna-jepa/eval_decision
LOG=/mnt/cunyuliu/rna-jepa/logs/rnaformer_bprna_new.log
DBN=$O/rnaformer_bprna_new.dbn
NPZ=$O/rnaformer_probs_bprna_new.npz
export PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval
export TMPDIR=/mnt/cunyuliu/tmp
cd /home/cunyuliu/rna-jepa || exit 1

free_gb() {
  CUDA_VISIBLE_DEVICES=$1 /home/cunyuliu/miniconda3/envs/editflow/bin/python -c \
    "import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)" 2>/dev/null
}

attempt=0
while [ ! -s "$DBN" ]; do
  attempt=$((attempt+1))
  if [ "$attempt" -gt 40 ]; then
    echo "[rnaf-new] FATAL: 40 attempts exhausted $(date '+%T')"; exit 1
  fi
  best=""; bestfree=0
  for c in 0 1 3 5; do
    fb=$(free_gb "$c")
    [ -z "$fb" ] && continue
    ok=$(awk -v x="$fb" -v n=10 'BEGIN{print (x>=n)?1:0}')
    if [ "$ok" = "1" ]; then
      if awk -v a="$fb" -v b="$bestfree" 'BEGIN{exit !(a>b)}'; then
        bestfree=$fb; best=$c
      fi
    fi
  done
  if [ -z "$best" ]; then
    echo "[rnaf-new] attempt $attempt: no card >=10GB free; wait $(date '+%T')" >&2
    sleep 120
    continue
  fi
  echo "[rnaf-new] attempt $attempt on GPU $best (${bestfree}GB) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$best PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY eval/ss/run_rnaformer.py \
    --state-dict $M/RNAformer_32M_state_dict_bprna.pth \
    --config $M/RNAformer_32M_config_bprna.yml \
    --split bprna_new --no-release-row \
    --out-dbn $DBN --out-npz $NPZ >> "$LOG" 2>&1
  if [ -s "$DBN" ]; then
    echo "[rnaf-new] DONE: $DBN $(date '+%T')"
    break
  fi
  echo "[rnaf-new] attempt $attempt died; retrying in 120s" >&2
  sleep 120
done
echo "[rnaf-new] ALL DONE $(date '+%T')"
