#!/bin/bash
# Round-28: mixed-corpus arm (GenF SOTA path per the three-control attribution).
# Corpus = bprna_tr1c (42,564, MFE teacher) + chanrg_train_train (95,181, GT
# teacher) -> single jsonl + merged teacher dir. Everything else fixed vs
# chanrg_r2d_s0_gt. Seed 0. Auto chain: build mixed store -> train 20k ->
# snapshot -> 4-split eval -> ALL DONE.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
LOG=/root/chanrg_arm_gt.log
J=/root/rna_jepa_data/ss_data/jsonl
E=/root/rna_jepa_data/embeddings/rinalmo-giga
D=/root/rna_jepa_data
OUT=$D/runs/chanrg_r2d_mix_s0
TEACHER=$D/ss_data/teacher_mix_slim

echo "[mix-s0] $(date) start" >> $LOG

# --- build merged corpus + teacher dir (idempotent) ---
if [ ! -f $TEACHER/manifest.json ]; then
  cat $J/bprna_tr1c.jsonl $J/chanrg_train_train.jsonl > $J/mixed_train.jsonl
  mkdir -p $TEACHER
  # copy all shards from both teacher dirs, then write a merged manifest
  cp $D/ss_data/teacher/bprna_tr1c/shard_*.npz $TEACHER/ 2>/dev/null
  cp $D/ss_data/teacher_gt/chanrg_train/shard_*.npz $TEACHER/ 2>/dev/null
  $PY - <<PYEOF
import json, os
T = "$TEACHER"
shards = []
for i, f in enumerate(sorted(os.listdir(T))):
    if f.endswith(".npz"):
        shards.append({"file": f, "index": i})
n_b = json.load(open("$D/ss_data/teacher/bprna_tr1c/manifest.json"))["n_sequences"]
n_c = json.load(open("$D/ss_data/teacher_gt/chanrg_train/manifest.json"))["n_sequences"]
json.dump({
    "meta": {"corpus": "$J/mixed_bprna_chanrg.jsonl",
             "corpus_records": n_b + n_c,
             "note": "mixed teacher: bprna_tr1c (thermo:viennarna MFE) + chanrg_train_train (GT consensus). Round-28 mix arm, follow-up of the three-control GenF attribution: board corpus complementarity is the remaining gap."},
    "teacher": "mixed:mfe+gt",
    "version_lock": "ViennaRNA 2.7.2 + Rfam consensus",
    "n_sequences": n_b + n_c,
    "shard_size": 256,
    "shards": shards,
}, open(os.path.join(T, "manifest.json"), "w"))
print("mixed teacher manifest:", len(shards), "shards", n_b + n_c, "labels")
PYEOF
  echo "[mix-s0] $(date) mixed store built" >> $LOG
fi

# EmbeddingStore alias entries (split=mixed_train -> existing shards) are
# pre-registered in the rinalmo-giga manifest; corpus jsonl is mixed_train.

N=$($PY -c "
import json
m = json.load(open('$E/manifest.json'))
sp = [e['split'] for e in m['entries']]
print(sp.count('mixed_train'))
" 2>/dev/null || echo 0)
if [ "$N" -lt 5 ]; then
  echo "[mix-s0] FATAL embedding alias entries missing ($N)" >> $LOG; exit 1
fi

cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m rnajepa.train_decision \
  --arm chanrg_r2d_mix_s0 \
  --out $OUT \
  --data $J/mixed_train.jsonl \
  --teacher-dir $TEACHER \
  --steps 20000 --batch-size 2 --lr 1e-4 --seed 0 \
  --log-every 25 --save-every 500 \
  --head flat --scorer resnet2d --gpu-reserve-gb 18 \
  --embedding-dir $E --embedding-d-model 1280 \
  --device cuda >> $LOG 2>&1
RC=$?
echo "[mix-s0] $(date) train rc=$RC" >> $LOG
[ $RC -ne 0 ] && exit 1

cp $OUT/resume.pt $D/ckpts_1014/chanrg_r2d_mix_s0_step20000.pt 2>/dev/null || true
echo "[mix-s0] $(date) snapshot frozen" >> $LOG

for SPLIT in chanrg_test chanrg_gena chanrg_genc chanrg_genf; do
  EVAL_OUT=$D/eval_decision/chanrg_mix_s0/$SPLIT
  mkdir -p $EVAL_OUT
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 $PY /home/rna_jev/eval/ss/evaluate_decision.py \
    --checkpoint $D/ckpts_1014/chanrg_r2d_mix_s0_step20000.pt \
    --data $J/$SPLIT.jsonl \
    --out $EVAL_OUT \
    --tag chanrg_mix_s0_$SPLIT \
    --prior-weight -1 \
    --calib-data $J/chanrg_train_validation.jsonl \
    --embedding-split $SPLIT \
    --head-chunk 8 \
    --device cuda >> $LOG 2>&1
  echo "[mix-s0] $(date) eval $SPLIT rc=$?" >> $LOG
done
echo "[mix-s0] $(date) ALL DONE" >> $LOG
