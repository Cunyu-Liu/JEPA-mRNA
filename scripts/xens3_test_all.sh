#!/bin/bash
# xens3 on the 8 test splits (w=0.7 VL0-locked; splits with embeddings;
# sequential solo runs on card 5 — two concurrent runs OOM'd, see 15.27).
# Idempotent per split: skips when result.json exists.
set -u
cd /home/cunyuliu/rna-jepa || exit 1
export PYTHONPATH=/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/tools:/mnt/cunyuliu/pylibs
EPY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
D=/mnt/cunyuliu/rna-jepa
CK0=$D/ckpts/plana_giga_tr1c_s0_step20000.pt
CK1=$D/ckpts/plana_giga_tr1c_s1_step20000.pt
CK2=$D/ckpts/plana_giga_tr1c_s2_step20000.pt
R2D=$D/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt
LOG=$D/logs/xens3_test_all.out

declare -A SPLITS=(
  [bprna_ts0]=bprna_ts0
  [bprna_new]=bprna_new
  [ref_pdb_ts1]=ref_pdb_ts1
  [ref_pdb_ts2]=ref_pdb_ts2
  [ref_pdb_ts3]=ref_pdb_ts3
  [ref_pdb_ts_hard]=ref_pdb_ts_hard
  [testsetb]=testsetb
  [archiveii_embok_clean]=archiveii_embok_clean
)

for sp in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean; do
  out=$D/eval_decision/xens3_${sp}_w0.7
  if [ -f "$out/result.json" ]; then
    echo "[xens3] SKIP $sp (result exists)" >> "$LOG"
    continue
  fi
  echo "[xens3] RUN $sp $(date '+%T')" >> "$LOG"
  env CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nice -n 15 "$EPY" tools/xens3_eval.py \
    --plana-ckpt0 $CK0 --plana-ckpt1 $CK1 --plana-ckpt2 $CK2 \
    --r2d-ckpt $R2D \
    --data $D/ss_data/jsonl/${sp}.jsonl \
    --embedding-split ${sp} \
    --out $out --w-plana 0.7 >> "$LOG" 2>&1
  echo "[xens3] $sp rc=$? $(date '+%T')" >> "$LOG"
done
echo "[xens3] ALL DONE $(date '+%F %T')" >> "$LOG"
