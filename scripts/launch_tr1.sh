#!/bin/bash
# Launch decision-model arms on the expanded corpus TR1.
#
# Why this script exists rather than reusing launch_ff_seeds.sh
# ------------------------------------------------------------
# The point of TR1 is a *controlled* comparison: TR0 (10,682 sequences) caused
# the headline number, so the arm that produced the headline must be re-run with
# everything identical except the corpus.  That means the same arm name
# (`rinalmo_ff`), the same flag set including the same `--nll-normalization`
# default that the headline run used, and only `--data`/`--teacher-dir` changed.
# Adding `--nll-normalization length` to this arm would confound the data change
# with an objective change, which is the one thing this run is supposed to
# isolate.
#
# Device selection: measured, not assumed.  On 2026-09-24 the node is full --
# GPU 0-5 hold 1-6 GiB free each and GPU 6/7 carry 12 and 13 processes over their
# MIG instances.  `nvidia-smi --query-compute-apps` reports the *parent* UUID even
# for MIG processes, so no per-slice figure can be derived from it.  The one
# solid measurement is the observed footprint of each arm:
#
#   rinalmo_* (frozen embeddings + head)   700-1500 MiB   (measured: 712, 752, 1456)
#   rinalmo_ff / rinalmo_sum               4890 MiB
#   full / nllonly (35M encoder, trained)  3362 MiB
#
# so only the small-footprint arms are launched here, onto GPU 7's two 3g.20gb
# slices, and the rest are left to the dispatch monitor.  Launching a 3.3 GiB
# `full` arm onto a slice with ~1.3 GiB free would OOM a shared card, which has
# already happened once on this project.
#
# Usage: bash scripts/launch_tr1.sh [--dry-run]
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
PY="${RNAJEV_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"
DATA="$ART/ss_data/jsonl/bprna_tr1.jsonl"
TEACH="$ART/ss_data/teacher/bprna_tr1"
EMB="$ART/embeddings/rinalmo-giga"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

export PYTHONPATH=/mnt/cunyuliu/pylibs:src
export TMPDIR=/mnt/cunyuliu/tmp

# preconditions: refuse rather than train on a half-built corpus or a teacher set
# that cannot satisfy strict lookup
if [ ! -s "$DATA" ]; then echo "FATAL: no corpus at $DATA" >&2; exit 2; fi
if [ ! -s "$TEACH/manifest.json" ]; then
  echo "FATAL: teacher manifest missing at $TEACH/manifest.json. Training with" >&2
  echo "       --teacher-dir is strict: a missing label aborts the run rather" >&2
  echo "       than falling back to the mock teacher." >&2
  exit 2
fi
n_corpus=$("$PY" -c "print(sum(1 for _ in open('$DATA')))")
n_teach=$("$PY" -c "import json;print(json.load(open('$TEACH/manifest.json'))['n_sequences'])")
echo "[pre] corpus=$n_corpus teacher_labels=$n_teach"
if [ "$n_corpus" != "$n_teach" ]; then
  echo "FATAL: corpus has $n_corpus records but the teacher set has $n_teach labels." >&2
  exit 2
fi

# Target MIG instances on GPU 7 (3g.20gb each).  UUIDs, not the parent index:
# setting the parent index leaves it ambiguous which instance is used.
DEV_MAIN="MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
DEV_ALT="MIG-10b9b777-a776-56de-9f7c-efde6c584f71"

launch() {  # launch <tag> <device> <extra-args...>
  local tag="$1" dev="$2"; shift 2
  local out="$ART/runs/$tag"
  # idempotency: a live process for this tag beats any checkpoint bookkeeping,
  # because a relaunch loop can leave an up-to-date resume.pt behind
  if pgrep -f "[r]uns/$tag" >/dev/null 2>&1; then
    echo "[skip] $tag already running"; return 0
  fi
  echo "[launch] $tag -> $dev"
  if [ "$DRY" -eq 1 ]; then
    echo "    CUDA_VISIBLE_DEVICES=$dev $PY -m rnajepa.train_decision --out $out $*"
    return 0
  fi
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" -m rnajepa.train_decision \
      --out "$out" --data "$DATA" --teacher-dir "$TEACH" \
      "$@" --device cuda --resume "$out/resume.pt" \
      > "$ART/runs/$tag.log" 2>&1 < /dev/null &
  echo "[launch] $tag pid=$!"
}

# 1. headline arm, corpus swapped and nothing else changed (see header).
#    `--head-chunk-size 16` rather than the headline run's 64: head chunking is
#    numerically equivalent, not an approximation -- tests/test_head_chunking.py
#    asserts the chunked and whole-matrix paths are bit-identical in forward and
#    agree to 1e-15 in float64 backward -- so lowering it changes peak memory and
#    step time but not the function being optimised, and therefore does not
#    confound the corpus comparison this arm exists to make.
launch rinalmo_ff_tr1_b4_s0 "$DEV_MAIN" \
    --arm rinalmo_ff --steps 20000 --batch-size 4 --lr 1e-4 --encoder-size 35M \
    --seed 0 --log-every 25 --save-every 500 \
    --embedding-dir "$EMB" --embedding-d-model 1280 --head-chunk-size 16

# 2. the length-normalised twin, so TR1 carries the same objective comparison
#    TR0 has, instead of only the sum-normalised side
launch rinalmo_len_tr1_b4_s0 "$DEV_ALT" \
    --arm rinalmo_len --steps 20000 --batch-size 4 --lr 1e-4 --encoder-size 35M \
    --seed 0 --log-every 25 --save-every 500 \
    --embedding-dir "$EMB" --embedding-d-model 1280 --head-chunk-size 16 \
    --nll-normalization length --lambda-nll 1 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0

echo "### TR1 launches submitted ($(date +%H:%M:%S))"