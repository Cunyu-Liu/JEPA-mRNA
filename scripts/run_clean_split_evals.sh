#!/bin/bash
# Re-measure on the de-duplicated splits.
#
# Why
# ---
# `tools/dedup_against_train.py` measured the training-set overlap of every split:
#
#   split        total   exact dup      homology>0.5   kept
#   ArchiveII    3950    812 (20.6%)    594 (15.0%)    2544 (64.4%)
#   TS0          1288      0  (0.0%)      3  (0.2%)    1285 (99.8%)
#   VL0           196      0  (0.0%)      0  (0.0%)     196 (100%)
#   bpRNA-new    5388      0  (0.0%)      0  (0.0%)    5388 (100%)
#
# ArchiveII's 5S rRNA rows are literally in TR0 (`5s_Acholeplasma-laidlawii-2.bpseq`
# is byte-identical to `bpRNA_RFAM_654.bpseq`), so 35.6% of that split had to go.
# Every ArchiveII number measured so far -- including the 0.8656 that beat centroid
# on the <=100 nt bucket -- is therefore a partly-memorised score and must be
# replaced by the number on the clean split.  The other three splits barely move,
# which is itself worth confirming rather than assuming.
#
# This queue waits for the four existing evaluation queues to finish so it cannot
# race them; the node is shared and the evaluator is effectively single-core.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src:/home/cunyuliu/rna-jepa/eval
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
JSONL=$ART/ss_data/jsonl
DEV=$JSONL/bprna_vl0.jsonl
LOG=$ART/eval_decision/clean_splits.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"
CKPT=$ART/ckpts/rinalmo_ff_ff_w05_snapshot.pt

wait_for_others() {
    local waited=0
    while [ "$waited" -lt 1200 ]; do
        if pgrep -f '[e]valuate_decision.py' > /dev/null \
        || pgrep -f '[r]un_followup_evals.sh' > /dev/null \
        || pgrep -f '[r]un_objective_2x2_eval.sh' > /dev/null \
        || pgrep -f '[r]un_step_trend_eval.sh' > /dev/null \
        || pgrep -f '[r]un_objective_2x2_step4000_eval.sh' > /dev/null \
        || pgrep -f '[l]ength_bucketed_leaderboard.py' > /dev/null; then
            sleep 30
            waited=$((waited + 1))
            continue
        fi
        return 0
    done
    return 1
}

run_eval() {  # data_file  w  tag  embedding_split
    local data="$1"; local w="$2"; local tag="$3"; local emb="${4:-}"
    local extra=()
    [ -n "$emb" ] && extra=(--embedding-split "$emb")
    echo "=== $tag on $data (w=$w) $(date '+%T') ==="
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 15 \
        "$PY" "$REPO/eval/ss/evaluate_decision.py" \
        --checkpoint "$CKPT" \
        --data "$JSONL/$data" \
        --calib-data "$DEV" \
        --prior-weight "$w" \
        "${extra[@]}" \
        --out "$ART/eval_decision/${tag}_$(basename "$data" .jsonl)" \
        --encoder-size 35M --device cuda --head-chunk 8 \
        --tag "${tag}_$(basename "$data" .jsonl)"
    echo "=== $tag done rc=$? $(date '+%T') ==="
}

{
  echo "waiting for the other queues $(date '+%T')"
  if ! wait_for_others; then
      echo "FATAL: still busy after 10 h; aborting rather than racing"
      exit 1
  fi
  echo "clear to run $(date '+%T')"

  # ArchiveII is the split that actually changed, so it gets both decode conventions.
  run_eval archiveii_embok_clean.jsonl -1 clean_ff3500 archiveii
  run_eval archiveii_embok_clean.jsonl 0.75 clean_ff3500w archiveii

  # TS0 loses only 3 rows; re-measuring confirms the headline is not resting on them.
  run_eval bprna_ts0_clean.jsonl -1 clean_ff3500

  # Length buckets for the clean ArchiveII, so the "beats centroid below 200 nt"
  # claim can be re-tested on de-duplicated data.
  echo "=== clean ArchiveII length buckets $(date '+%T') ==="
  nice -n 15 "$PY" "$REPO/tools/length_bucketed_leaderboard.py" \
      --repo "$REPO" \
      --eval-root "$ART/eval_decision" \
      --data-dir "$JSONL" \
      --split archiveii_embok_clean \
      --run clean_ff3500w_archiveii_embok_clean \
      --bucket 100 --bucket 200 --bucket 400 \
      --json "$ART/eval_decision/buckets_archiveii_clean.json"
  echo "=== buckets done rc=$? $(date '+%T') ==="

  echo "########## clean-split re-measurement done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
