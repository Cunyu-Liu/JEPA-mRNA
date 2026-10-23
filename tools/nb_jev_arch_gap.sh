#!/bin/bash
# nb_jev archiveii gap-fill v2: truncated eval needs its own embedding shard.
# extract_nb_embeddings.py --split <name> reads $JSONL/<name>.jsonl; the trunc598
# jsonl already exists. Extract, then eval. (Embedding rows for truncated seqs
# are prefix-consistent with the full shard but the store is keyed by seq string.)
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
J=/root/rna_jepa_data/ss_data/jsonl
LOG=/root/nb_jev.log
export NB_CKPT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt
export PYTHONPATH=/root/rna_jepa_data/hf_home/nucleicbert-code:/home/rna_jev/tools:/home/rna_jev/src

cd /home/rna_jev/tools
if [ ! -f /root/rna_jepa_data/embeddings/nucleicbert/archiveii_embok_clean_trunc598.shard0of1.npz ]; then
  $PY extract_nb_embeddings.py --split archiveii_embok_clean_trunc598 --batch-size 8 >> $LOG 2>&1
  echo "[ta99-gap] $(date) emb trunc598 rc=$?" >> $LOG
fi

EVAL_OUT=/root/rna_jepa_data/eval_decision/nb_jev_r2dtr1c_s0_step20000/archiveii_embok_clean_trunc598
mkdir -p $EVAL_OUT
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
$PY /home/rna_jev/eval/ss/evaluate_decision.py \
  --checkpoint /root/rna_jepa_data/runs/nb_jev_r2dtr1c_s0/resume.pt \
  --data $J/archiveii_embok_clean_trunc598.jsonl \
  --out $EVAL_OUT \
  --tag nb_jev_archiveii_trunc598 \
  --prior-weight -1 \
  --calib-data $J/bprna_vl0.jsonl \
  --head-chunk 8 \
  --device cuda >> $LOG 2>&1
echo "[ta99-gap] $(date) archiveii trunc598 eval rc=$?" >> $LOG
