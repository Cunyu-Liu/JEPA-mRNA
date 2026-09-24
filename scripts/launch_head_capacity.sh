#!/bin/bash
# Head-capacity arm: is the 519k-parameter head the ceiling on F1?
#
# The flat head over frozen RiNALMo-giga embeddings has only **519,503 trainable
# parameters** at the default d_z=128 / hidden=64 (measured, not estimated), of which
# PairRepresentation.proj = Linear(3*1280, 128) dominates.  Everything else about the
# best arm (`rinalmo_ff_b4_s0`) is frozen: same data, teacher, backbone, batch, lr,
# steps and seed.  So this arm changes exactly one thing -- capacity:
#
#   d_z 128 -> 512, hidden 64 -> 512   =>   2,507,791 trainable parameters (4.8x)
#
# Why this is the right next experiment rather than more epochs
# ------------------------------------------------------------
# The model is at ~1.4 epochs of TR0 and its TS0 F1 has been flat since step 2000
# (0.4554 @1000, 0.4861 @2000, 0.4778 @3500 at a fixed decode weight), while its
# cross-family F1 falls to 0.3094 -- within 0.008 of the *untrained-prior* baseline
# (Nussinov+Turner 0.3015).  A model that has stopped improving in-distribution and
# collapses out-of-family is the signature of too little capacity for the signal, not
# of too little data (the extra corpora are unusable: RNAStrAlign_bpseq is all zero
# bytes in this extraction, and Rfam is reserved as the family-level test split).
#
# The objective is the length-normalised one, so this arm is comparable with
# `rinalmo_len_b4_s0` (same objective, default capacity) as well as with `rinalmo_ff`.
#
# Head chunk is 16, not 64: the chunk transient is (B, L, chunk, 3*d_model) =
# (4, 500, chunk, 3840) float32, i.e. ~0.5 GiB at chunk 16 and ~2 GiB at chunk 64.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
D=/mnt/cunyuliu/rna-jepa
DATA=$D/ss_data/jsonl/bprna_tr0.jsonl
TEACHER=$D/ss_data/teacher/bprna_tr0
EMB=$D/embeddings/rinalmo-giga
cd /home/cunyuliu/rna-jepa || exit 1

DEV="${RNAJEV_BIG_GPU:?set RNAJEV_BIG_GPU to a MIG uuid or GPU index with >= 6 GiB free}"
TAG="${RNAJEV_BIG_TAG:-rinalmo_big_b4_s0}"
DZ="${RNAJEV_BIG_DZ:-512}"
HID="${RNAJEV_BIG_HIDDEN:-512}"
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log

if [ -e "$OUT/resume.pt" ]; then
  echo "SKIP $TAG (already has a checkpoint at $OUT/resume.pt)"
  exit 0
fi

echo "=== launch $TAG on $DEV d_z=$DZ hidden=$HID $(date '+%T') -> $LOG ==="
CUDA_VISIBLE_DEVICES="$DEV" setsid nohup "$PY" -m rnajepa.train_decision \
    --arm "$TAG" --out "$OUT" \
    --data "$DATA" --steps 20000 --batch-size 4 --lr 1e-4 \
    --encoder-size 35M --seed 0 --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --d-z "$DZ" --hidden "$HID" --head-chunk-size 16 \
    --nll-normalization length \
    --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 \
    > "$LOG" 2>&1 < /dev/null &

echo "=== launched; sleeping 150s then reporting ==="
sleep 150
grep -aE "step |FATAL|Error|out of memory" "$LOG" | tail -4
grep -a "n_learnable_params" "$OUT/run_meta.json" 2>/dev/null || true
