#!/bin/bash
# Relaunch a single failed seed arm, pinned to an *isolated* MIG instance.
#
# Why not just reuse scripts/launch_ff_seeds.sh
# ---------------------------------------------
# That script placed the seed arms on shared full cards (GPU 3/4/5), measured free at
# 27.2/16.7/8.0 GiB at launch time.  `rinalmo_ff_b4_s1` then died with
#
#   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.19 GiB.
#   GPU 0 has a total capacity of 39.49 GiB of which 1.10 GiB is free.
#
# GPU 4's headroom had been consumed by other users' jobs between the measurement and
# the first long-sequence batch.  Free memory on a shared card is a *transient*
# observation, not a reservation; a MIG instance is isolated and therefore holds its
# headroom.  The 10-minute monitor caught the death (16:40:36 alert: traceback, no live
# process) rather than letting it fail silently, which is what that cron exists for.
#
# So this script pins by MIG UUID and re-checks the slice's free memory immediately
# before launching.  Configuration is identical to the surviving seed arms (chunk 64,
# default lambdas, same data/teacher/backbone/batch/lr/steps); only the seed and the
# device differ, and the device is not part of the scientific configuration.
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

TAG="${1:-rinalmo_ff_b4_s1}"
SEED="${2:-1}"
DEV="${3:-MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585}"
MIN_FREE_MIB="${4:-3000}"

# Idempotency: refuse to start a second copy of the same arm.
#
# `resume.pt` alone is NOT a sufficient check.  A fresh run spends several minutes
# loading the frozen embedding store before it writes any checkpoint, so during that
# window the file does not exist and a second launch looks legitimate.  That is exactly
# how running this script twice started two `rinalmo_ff_b4_s1` processes writing to the
# same output directory and the same log -- caught by inspection, not by the script.
# So the primary check is "is a process already running with this --out directory",
# and `resume.pt` is only a secondary check.
if pgrep -f "rnajepa.train_decision.*--out $D/runs/$TAG\b" > /dev/null; then
  echo "SKIP $TAG: a training process is already running with --out $D/runs/$TAG"
  exit 0
fi
if [ -e "$D/runs/$TAG/resume.pt" ]; then
  echo "SKIP $TAG: a checkpoint already exists at $D/runs/$TAG/resume.pt"
  exit 0
fi

# Refuse to launch into a slice that is too full -- the failure this script exists to
# avoid.
#
# `nvidia-smi --query-gpu=memory.free -i <MIG-UUID>` does NOT work: it prints
# "No devices were found", and the first version of this guard compared that string
# against an integer, failed the test, and launched anyway -- a guard that silently
# does nothing is worse than no guard.  So the lookup goes through the repo's own
# `gpu_util.sh`, which already knows how to enumerate MIG instances and their free
# memory, and a missing target is a hard failure rather than a pass.
source "$(dirname "$0")/gpu_util.sh"

free="$(list_schedulable_gpus | awk -v t="$DEV" '$1 == t {print $2}')"
if [ -z "$free" ]; then
  echo "FATAL: $DEV is not among the schedulable targets -- refusing to launch."
  echo "       (a target that cannot be measured must not be treated as having room)"
  list_schedulable_gpus
  exit 2
fi
echo "target $DEV has ${free} MiB free (need >= ${MIN_FREE_MIB})"
if [ "$free" -lt "$MIN_FREE_MIB" ]; then
  echo "REFUSING to launch $TAG: only ${free} MiB free on $DEV"
  exit 3
fi

echo "=== launch $TAG on $DEV seed=$SEED $(date '+%T') ==="
CUDA_VISIBLE_DEVICES="$DEV" setsid nohup nice -n 5 "$PY" -m rnajepa.train_decision \
    --arm "$TAG" --out "$D/runs/$TAG" \
    --data "$DATA" --steps 20000 --batch-size 4 --lr 1e-4 \
    --encoder-size 35M --seed "$SEED" --device cuda \
    --log-every 25 --save-every 500 \
    --teacher-dir "$TEACHER" \
    --embedding-dir "$EMB" --embedding-d-model 1280 \
    --head-chunk-size 64 \
    > "$D/runs/$TAG.log" 2>&1 < /dev/null &

sleep 5
echo "detached; log -> $D/runs/$TAG.log"
