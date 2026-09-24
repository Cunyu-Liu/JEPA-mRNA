#!/bin/bash
# Seed replicates of the headline arm.
#
# Why
# ---
# Every number in the draft comes from ONE seed (`rinalmo_ff_b4_s0`, seed 0).  The
# frozen protocol (spec/benchmark_decision.md 4.3) requires >=5 seeds and a mean +- std,
# and the draft currently has to list "single seed for the headline" as a limitation.
# These arms are the exact headline configuration -- same data, teacher, backbone,
# batch, lr, steps, head size, chunking, and default lambdas (all 1.0, i.e.
# nll_normalization=sum with all four objective terms on) -- with only the seed
# changed.  Nothing else varies, so the spread across them is seed variance and
# nothing else.
#
# Device placement
# ----------------
# Free memory measured immediately before launching (nvidia-smi):
#   GPU 3  8029 MiB   GPU 4  27206 MiB   GPU 5  16672 MiB
# A head-only arm with chunk 64 peaks near 1.3 GiB, so each card has room.  The cards
# are shared, so the arms are pinned by parent index and run under `nice`; they yield
# to other users rather than competing with them.
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
  local tag="$1"; local dev="$2"; local seed="$3"
  local out="$D/runs/$tag"
  local log="$D/runs/$tag.log"
  if [ -e "$out/resume.pt" ]; then
    echo "SKIP $tag (already has a checkpoint at $out/resume.pt)"
    return
  fi
  echo "=== launch $tag on device $dev seed=$seed $(date '+%T') -> $log ==="
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision \
      --arm "$tag" --out "$out" \
      --data "$DATA" --steps 20000 --batch-size 4 --lr 1e-4 \
      --encoder-size 35M --seed "$seed" --device cuda \
      --log-every 25 --save-every 500 \
      --teacher-dir "$TEACHER" \
      --embedding-dir "$EMB" --embedding-d-model 1280 \
      --head-chunk-size 64 \
      > "$log" 2>&1 < /dev/null &
  sleep 4
}

launch rinalmo_ff_b4_s1 "4" 1
launch rinalmo_ff_b4_s2 "5" 2
launch rinalmo_ff_b4_s3 "3" 3

echo "detached; logs under $D/runs/rinalmo_ff_b4_s{1,2,3}.log"
