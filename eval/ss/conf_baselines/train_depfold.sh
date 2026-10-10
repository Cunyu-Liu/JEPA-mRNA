# [archived] DEPfold (ICLR 2025) TR0 retrain script (official protocol, RNA-FM t12).
# Source of truth: /home/cunyuliu/rna_baselines_src/DEPfold_data/train_depfold.sh, run 2026-10-10.
# Early-stop epoch 41, VL0 F1 0.6675. ckpt: DEPfold_data/depfold_bpRNA_rnafm/model.pt.
# Upstream bug patched (one line): DEPfold/models/biaffine_supar.py:232
#   contact_map broadcast [B,L,L]*[B,L] -> mask.unsqueeze(-1) (semantics unchanged).
# Board JSON (NOT committed): /mnt/cunyuliu/rna-jepa/eval_decision/depfold_board.json
#!/bin/bash
# T-A49: DEPfold (ICLR 2025) baseline — train on bpRNA TR0 (their protocol),
# then predict on our board splits. Idempotent; logs to DEPfold_data/.
# GPU pick: most-free of 0-5 at launch (user rule: use any VRAM available).
set -u
cd /home/cunyuliu/rna_baselines_src/DEPfold

PY="/home/cunyuliu/miniconda3/envs/lucaone/bin/python"
export PYTHONPATH=/mnt/cunyuliu/pylibs
DATA=/home/cunyuliu/rna_baselines_src/DEPfold_data
LOGDIR=/home/cunyuliu/rna_baselines_src/DEPfold_data
mkdir -p "$LOGDIR"

# pick most-free card among 0-5
gpu=1; bestfree=0
for c in 0 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $c 2>/dev/null) || continue
  free=$(( t - u ))
  if [ "$free" -gt "$bestfree" ]; then bestfree=$free; gpu=$c; fi
done
echo "[depfold] using GPU $gpu (${bestfree}MB free) $(date '+%F %T')"

MODEL_OUT=$LOGDIR/depfold_bpRNA_rnafm
if [ ! -f "$MODEL_OUT/model.pt" ]; then
  # train: RNA-FM embedding, TR0 train, VL0 eval (their early-stop protocol)
  CUDA_VISIBLE_DEVICES=$gpu $PY run_parser.py \
    --mode train \
    --train_path $DATA/TR0 \
    --eval_path /home/cunyuliu/rna_baselines_src/DEPfold_data/VL0 \
    --test_path $DATA/TS0 \
    --train_session TR0 --eval_session VL0dep --test_session TS0dep \
    --embedding RNA-fm \
    --output_dir $MODEL_OUT \
    --cache_data ./data/depfold_run_ \
    > $LOGDIR/depfold_train.log 2>&1
  echo "[depfold] train rc=$?"
else
  echo "[depfold] model already trained"
fi
echo "[depfold] DONE $(date '+%F %T')"
