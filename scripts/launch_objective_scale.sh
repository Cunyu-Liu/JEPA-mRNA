#!/bin/bash
# Launch the 2x2 objective-scaling experiment (2026-09-24).
#
# Why this experiment exists
# --------------------------
# Measured on real batches (4-step real-data pre-flight, bprna_tr0, batch 4):
#
#   nll_normalization=sum     nll 31-88    distill 1.1-1.3  rlcd -0.06..-0.09  cal 1.1-1.3
#   nll_normalization=length  nll 0.40-0.71 distill 1.1-1.3 rlcd -0.06..-0.09  cal 1.1-1.3
#
# With the historical "sum" the CRF term is O(L) while the three auxiliary terms are
# per-pair means, so with lambda_* = 1 the auxiliary terms are ~4% of the objective:
# the arms `full` and `nllonly` were numerically the same experiment, and the H6
# ablation (does RLCD/distillation do anything?) could not be answered at all.
# "length" divides the CRF term by L so the four terms are the same order of
# magnitude and lambda = 1 means what the spec says it means.
#
# The design is the full 2x2 over {sum, length} x {aux on, aux off} on the *same*
# frozen-RiNALMo head, so the two effects are separable:
#
#   sum    + aux off : rinalmo_sum_b4_s0   (new; the historical objective as a control)
#   sum    + aux on  : rinalmo_ff_b4_s0    (already running, 20000 steps)
#   length + aux off : rinalmo_len_b4_s0   (new)
#   length + aux on  : rinalmo_bal_b4_s0   (new) + rinalmo_bal_b4_s1 (new, seed variance)
#
# Everything else (data, teacher, backbone, batch, lr, steps) is identical to
# rinalmo_ff_b4_s0, so any difference is attributable to the objective.
#
# Device placement is pinned by MIG UUID, not by parent index: the node is shared
# and setting the parent index leaves it undefined which MIG instance you get.
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

launch() {
  local tag="$1"; local dev="$2"; local chunk="$3"; local seed="$4"; shift 4
  local out="$D/runs/$tag"
  local log="$D/runs/$tag.log"
  if [ -e "$out/resume.pt" ]; then
    echo "SKIP $tag (already has a checkpoint at $out/resume.pt)"
    return
  fi
  echo "=== launch $tag on $dev chunk=$chunk seed=$seed $(date '+%T') -> $log ==="
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" -m rnajepa.train_decision \
      --arm "$tag" --out "$out" \
      --data "$DATA" --steps 20000 --batch-size 4 --lr 1e-4 \
      --encoder-size 35M --seed "$seed" --device cuda \
      --log-every 25 --save-every 500 \
      --teacher-dir "$TEACHER" \
      --embedding-dir "$EMB" --embedding-d-model 1280 \
      --head-chunk-size "$chunk" \
      "$@" > "$log" 2>&1 < /dev/null &
  sleep 3
}

BIG="MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"

# Free memory measured immediately before this run (nvidia-smi / tools/mig_free.py):
#   MIG-6e59f9af (3g.20gb, GPU 7) 14910 MiB free   <- the only roomy MIG instance
#   MIG-10b9b777 (3g.20gb, GPU 7)  1913 MiB free   <- too small for batch 4 now
#   GPU 3 10600 MiB free, GPU 4 13200 MiB free     <- shared, but genuinely free
# So two arms go on the one roomy MIG slice and two on the shared cards with real
# headroom.  A head-only arm with chunk 16 peaks around 1.5 GiB, so ~2 GiB each.
launch rinalmo_bal_b4_s0 "$BIG" 64 0 \
    --nll-normalization length --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1
launch rinalmo_bal_b4_s1 "$BIG" 16 1 \
    --nll-normalization length --lambda-nll 1 --lambda-distill 1 --lambda-rlcd 1 --lambda-cal 1
launch rinalmo_len_b4_s0 "4"    16 0 \
    --nll-normalization length --lambda-nll 1 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0
launch rinalmo_sum_b4_s0 "3"    16 0 \
    --nll-normalization sum --lambda-nll 1 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0

echo "=== launched; sleeping 90s then reporting ==="
sleep 90
for t in rinalmo_bal_b4_s0 rinalmo_bal_b4_s1 rinalmo_len_b4_s0 rinalmo_sum_b4_s0; do
  echo "--- $t"; tail -2 "$D/runs/$t.log" 2>&1
done
