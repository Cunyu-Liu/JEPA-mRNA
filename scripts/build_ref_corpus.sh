#!/bin/bash
# Build corpora from the RNAformer reference release (the .plk DataFrames).
#
# Why this exists
# ---------------
# The first inventory of training data looked only at /mnt/cunyuliu/BPfold_data
# and concluded the project had 10,682 training sequences and two usable test
# splits.  That was wrong, and the correction matters:
#
#   * training  10,682 -> 90,579 real sequences (8.5x) and, separately, 410,408
#               synthetic sequences whose structures are by construction
#               canonical + nested (exactly the space the decision head emits);
#   * benchmark 2 splits -> 9 (pdb ts1/ts2/ts3/ts_hard, synthetic test,
#               bprna_ts0, synthetic valid, plus the .bpseq sets already built).
#
# The release is the one RNAformer ships, so numbers produced on it are
# *directly* comparable to published RNAformer figures rather than merely
# similar-looking.
#
# Policy, and why each flag is not a default
# ------------------------------------------
# --pseudoknot-source pk    the release labels every pair with its crossing
#                           class, so pseudoknotted pairs can be removed exactly
#                           instead of by the greedy heuristic that also removes
#                           innocent nested pairs.
# --pair-type-policy canonical-drop
#                           non-canonical pairs are 9.1% of `bprna_data` train
#                           and 21.8% of `pdb_ts1`, against 0.0% in the corpora
#                           built from .bpseq.  A head that can only emit
#                           AU/GC/GU must not be scored against pairs it cannot
#                           express.  Both raw and kept counts are in the
#                           manifest so the removal is auditable.
# --multiplet-policy drop   one base annotated with several partners.  The
#                           shared-left form is caught downstream as a crossing
#                           but resolved by degree; the shared-right form is not
#                           caught at all and would be written as an unbalanced
#                           dot-bracket (see tests/test_ss_data_prep.py).
# --structure-crosscheck report
#                           NOT reject: the release's `structure` column is
#                           provably mangled on pseudoknot-heavy rows
#                           (`pdb_ts1` Id 632970: 37 index pairs vs 29 bracket
#                           pairs), so rejecting on it would discard exactly the
#                           hardest structures the benchmark exists to measure.
#
# Usage: bash scripts/build_ref_corpus.sh [output_dir] [--benchmark-only|--train-only]
set -uo pipefail

PY="${RNAJEV_PYTHON:-$HOME/miniconda3/envs/lucaone/bin/python}"
OUT="${1:-/mnt/cunyuliu/rna-jepa/ss_data/jsonl}"
MODE="${2:-all}"
D="${RNAJEV_REF_DATASETS:-/mnt/cunyuliu/rna-jepa/refmodels/datasets}"

mkdir -p "$OUT"
AUDIT="$OUT/_ref_content_audit.txt"
: >"$AUDIT"

POLICY=(--min-loop-policy drop --pseudoknot-policy drop
        --pseudoknot-source pk --pair-type-policy canonical-drop
        --multiplet-policy drop --structure-crosscheck report)

# ---- 0. content audit: a .plk that is missing or 0 bytes is a hard failure ---
FAILED=0
check_plk() {
  local label="$1" file="$2" required="$3"
  if [ ! -s "$file" ]; then
    printf '%-24s MISSING_OR_EMPTY  %s\n' "$label" "$file" | tee -a "$AUDIT"
    if [ "$required" -eq 1 ]; then
      echo "FATAL: required .plk '$label' missing/empty at $file" >&2
      FAILED=1
    fi
  else
    printf '%-24s bytes=%-12d %s\n' "$label" "$(stat -c%s "$file")" "$file" | tee -a "$AUDIT"
  fi
}

echo "=== .plk content audit ($(date +%H:%M:%S)) ==="
check_plk test_sets               "$D/test_sets.plk"                        1
check_plk biophysical_model_data  "$D/biophysical_model_data.plk"           0
check_plk bprna_data              "$D/bprna_data.plk"                      0
check_plk experimental_pretrain   "$D/experimental_pretrain_data.plk"      0
check_plk intra_family            "$D/intra_family_experimental_data.plk"  0
check_plk inter_family            "$D/inter_family_experimental_data.plk"  0
[ "$FAILED" -eq 1 ] && exit 2

run_refplk() {  # run_refplk <name> <plk-file> <set-name>
  local name="$1" file="$2" setname="$3"
  echo "### $name  set=$setname  ($(date +%H:%M:%S))"
  "$PY" data/ss/prepare_decision_data.py \
      --ref-plk "$D/$file" --ref-plk-set "$setname" "${POLICY[@]}" \
      --out "$OUT/$name.jsonl" --manifest "$OUT/$name.manifest.json" 2>&1 | tail -24
  echo "--- $name rc=$?"
}

if [ "$MODE" != "--train-only" ]; then
  # ---- new benchmark splits ------------------------------------------------
  # pdb ts1/ts2/ts3 are the field-standard experimental-label sets and ts_hard
  # is the adversarial one; bprna_ts0 is the same split as our .bpseq TS0, which
  # makes it a free cross-check on two independent readers of one dataset.
  run_refplk ref_pdb_ts1         test_sets.plk pdb_ts1
  run_refplk ref_pdb_ts2         test_sets.plk pdb_ts2
  run_refplk ref_pdb_ts3         test_sets.plk pdb_ts3
  run_refplk ref_pdb_ts_hard     test_sets.plk pdb_ts_hard
  run_refplk ref_synthetic_test  test_sets.plk synthetic_test
  run_refplk ref_bprna_ts0       test_sets.plk bprna_ts0
  # synthetic_valid is the release's own held-out synthetic split.  It matters
  # because VL0 is unusable for selection (50.5% CRW vs TS0's 7.3%), and a
  # selection set with a different source mix than the test set selects for the
  # wrong thing.
  run_refplk ref_synthetic_valid biophysical_model_data.plk synthetic_valid
fi

if [ "$MODE" != "--benchmark-only" ]; then
  # ---- expanded training: real data ---------------------------------------
  run_refplk ref_tr_bprna        bprna_data.plk                  train
  run_refplk ref_tr_experimental experimental_pretrain_data.plk  train
  run_refplk ref_tr_intra        intra_family_experimental_data.plk train
  run_refplk ref_tr_inter        inter_family_experimental_data.plk train
  # ---- expanded training: large synthetic set ----------------------------
  # Kept in its own file so it can be included or excluded by one flag at
  # training time instead of being baked into the real-data corpus.
  run_refplk ref_syn_train       biophysical_model_data.plk       train
fi

echo "### ref corpus done ($(date +%H:%M:%S))"