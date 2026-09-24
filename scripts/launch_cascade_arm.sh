#!/bin/bash
# Launch the C2 (hierarchical cascade) training arm -- the first time the cascade has
# ever been on a training path.
#
# Why this is a new experiment and not a rerun
# -------------------------------------------
# The second-round audit (spec §0.9.5 finding A) established that the cascade was not
# a usable architecture:
#   * its L0 selection was a hard top_k, so nothing could teach it which block pairs
#     matter; measured untrained helix recall was 0.1548 against the P6 gate of 0.98,
#     and with k=2 the ceiling is structural, not a training problem;
#   * it wrote -inf on 94.14% of candidate pairs, so the CRF's log Z no longer
#     described a distribution the model could produce (flat_nll 6.271 vs
#     cascade_nll 90017.977 at L=60) -- it could not be trained with the project's own
#     objective at all.
#
# `rnajepa.cascade_objective` replaces both: a learned, differentiable gate
# `sigmoid((logit - theta)/tau)` supplies a finite `log gate` in place of -inf, and a
# layered objective (L0 block pairs / L1 helices / L2 base pairs / sparsity) gives each
# level its own term, normalised by its own decision-unit count.
#
# Everything except the head is identical to `rinalmo_ff_b4_s0` (same frozen RiNALMo
# embeddings, same data, same teacher, same batch/lr/steps/seed), so a difference is
# attributable to the head.
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

# Free memory measured immediately before launching (tools/mig_free.py):
#   MIG-10b9b777 (3g.20gb, GPU 7) 13382 MiB free  <- roomy enough for batch 4
#   MIG-6e59f9af (3g.20gb, GPU 7)  8576 MiB free
# The cascade materialises a dense (B, L, L) score matrix (the DP is O(L^3) dense in
# this implementation regardless), so its peak is comparable to the flat head's.
DEV="${RNAJEV_CASCADE_GPU:-MIG-10b9b777-a776-56de-9f7c-efde6c584f71}"
TAG=rinalmo_casc_b4_s0
OUT=$D/runs/$TAG
LOG=$D/runs/$TAG.log

if [ -e "$OUT/resume.pt" ]; then
  echo "SKIP $TAG (already has a checkpoint at $OUT/resume.pt)"
  exit 0
fi

echo "=== launch $TAG on $DEV $(date '+%T') -> $LOG ==="
CUDA_VISIBLE_DEVICES="$DEV" setsid nohup "$PY" -m rnajepa.train_decision \
    --arm "$TAG" --out "$OUT" \
    --data "$DATA" --steps 20000 --batch-size 4 --lr 1e-4 \
    --encoder-size 35M --seed 0 --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --head cascade --cascade-block-size 8 \
    --cascade-l0 1 --cascade-l1 1 --cascade-l2 1 --cascade-sparse 0.1 \
    --cascade-miss-cost 20 \
    --nll-normalization length \
    --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1 \
    > "$LOG" 2>&1 < /dev/null &

echo "=== launched; sleeping 120s then reporting ==="
sleep 120
tail -4 "$LOG"
