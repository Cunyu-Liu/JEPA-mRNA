#!/bin/bash
# Post-sweep evaluation queue: the ArchiveII number on the evaluable subset, then the
# VL0-independent ("as-trained") headline numbers.
#
# Two separate correctness problems are fixed here.
#
# 1. ArchiveII cannot be scored end-to-end with frozen embeddings.
#    RiNALMo-giga's `max_pos` is 1024, so 16 of ArchiveII's 3,966 rows (the ones longer
#    than that) have no cached embedding, and the evaluator refuses to fall back to the
#    encoder (correctly -- mixing frozen and computed representations would make the
#    result meaningless).  The scored split is therefore
#    `archiveii_embok.jsonl` = the 3,950 rows that do have embeddings, max length 954,
#    3,435 unique sequences.  `--embedding-split archiveii` points at the shard the
#    filtered view was derived from.  Any reported ArchiveII number must state this.
#
# 2. The headline must not depend on bpRNA VL0.
#    On the *same* length bucket (<=100 nt) the same checkpoint scores micro F1 0.9230 on
#    VL0 but 0.6188 on TS0 -- a 0.30 gap with no identified cause (exact overlap is zero;
#    K=20 containment against TR0 is 0.042 vs 0.040; length, pair density and family
#    labels are matched).  So the headline uses `--prior-weight -1`, i.e. the model's own
#    trained prior weight with no selection on any split, and the VL0-selected w is kept
#    only as a sensitivity analysis.
set -u

export PYTHONPATH=/mnt/cunyuliu/pylibs:/home/cunyuliu/rna-jepa/src
export TMPDIR=/mnt/cunyuliu/tmp
export OMP_NUM_THREADS=2

PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
REPO=/home/cunyuliu/rna-jepa
ART=/mnt/cunyuliu/rna-jepa
CKPT=$ART/ckpts/rinalmo_ff_ff_w05_snapshot.pt
DEV=$ART/ss_data/jsonl/bprna_vl0.jsonl
LOG=$ART/eval_decision/ood_followup.log
: > "$LOG"
EVAL_GPU="${RNAJEV_EVAL_GPU:-MIG-27707c52-3cf5-55f8-858f-1419d97bbdf3}"

run_eval() {  # data_file  w  tag  embedding_split
    local data="$1"; local w="$2"; local tag="$3"; local emb="${4:-}"
    local extra=()
    [ -n "$emb" ] && extra=(--embedding-split "$emb")
    echo "=== $tag on $(basename "$data") (w=$w) $(date '+%T') ==="
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        PYTHONPATH="$PYTHONPATH" TMPDIR="$TMPDIR" OMP_NUM_THREADS=2 nice -n 10 \
        "$PY" "$REPO/eval/ss/evaluate_decision.py" \
        --checkpoint "$CKPT" \
        --data "$ART/ss_data/jsonl/$data" \
        --calib-data "$DEV" \
        --prior-weight "$w" \
        "${extra[@]}" \
        --out "$ART/eval_decision/${tag}_$(basename "$data" .jsonl)" \
        --encoder-size 35M --device cuda --head-chunk 8 \
        --tag "${tag}_$(basename "$data" .jsonl)"
    echo "=== $tag done rc=$? $(date '+%T') ==="
}

{
  echo "waiting for the VL0-sweep evaluator to finish $(date '+%T')"
  for _ in $(seq 1 720); do
      pgrep -f '[e]valuate_decision.py' > /dev/null || break
      sleep 30
  done
  if pgrep -f '[e]valuate_decision.py' > /dev/null; then
      echo "FATAL: another evaluator is still running after 6 h; aborting rather than racing it"
      exit 1
  fi
  echo "clear to run $(date '+%T')"

  # 1. ArchiveII on the evaluable subset, at the VL0-selected weight.
  run_eval archiveii_embok.jsonl 0.75 sel_ff3500 archiveii

  # 2. VL0-independent headline: the model's own trained prior weight.
  for f in bprna_ts0.jsonl archiveii_embok.jsonl bprna_new.jsonl; do
      emb=""
      [ "$f" = "archiveii_embok.jsonl" ] && emb=archiveii
      run_eval "$f" -1 astrained_ff3500 "$emb"
  done
  echo "########## follow-up evals done $(date '+%T') ##########"
} >> "$LOG" 2>&1

echo "detached: log -> $LOG"
