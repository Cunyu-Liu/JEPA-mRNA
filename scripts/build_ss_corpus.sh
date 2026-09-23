#!/bin/bash
# Build the JSONL structure corpora for the decision model.
#
# Policy choice (see spec/benchmark_decision.md):
#   --min-loop-policy drop --pseudoknot-policy drop
# The decision head and the harness can only represent nested structures with
# hairpin loops of at least three, so the ground truth has to be projected into
# that same space for F1 to mean anything.  Rejecting instead of projecting
# would discard 4.6% of bpRNA TR0 and 71% of PDB ts3 -- and ts3 is part of the
# only experimental-label set we have.  Every removed pair is counted in the
# manifest, so a run can report "N pairs removed to legalise the ground truth".
#
# To get the strict-policy numbers as data-quality evidence, re-run a single set
# with the default policy; its reject_reasons are exactly the "records the field
# treats as ground truth that are not representable" statistic.
#
# Usage: bash scripts/build_ss_corpus.sh [output_dir]
set -euo pipefail

PY="${RNAJEV_PYTHON:-$HOME/miniconda3/envs/lucaone/bin/python}"
OUT="${1:-/mnt/cunyuliu/rna-jepa/ss_data/jsonl}"
B=/mnt/cunyuliu/BPfold_data
X=/mnt/cunyuliu/rna_ss_data/bpfold/extracted/BPfold_data
RINALMO=/mnt/cunyuliu/rna_ss_data/rinalmo

mkdir -p "$OUT"

POLICY=(--min-loop-policy drop --pseudoknot-policy drop)

run_bpseq() {
  local name="$1" dir="$2"
  echo "### $name  ($(date +%H:%M:%S))"
  "$PY" data/ss/prepare_decision_data.py \
      --bpseq-dir "$dir" "${POLICY[@]}" \
      --out "$OUT/$name.jsonl" --manifest "$OUT/$name.manifest.json" 2>&1 | tail -16
}

run_csv() {
  local name="$1" csv="$2"
  echo "### $name  ($(date +%H:%M:%S))"
  "$PY" data/ss/prepare_decision_data.py \
      --rinalmo-csv "$csv" "${POLICY[@]}" \
      --out "$OUT/$name.jsonl" --manifest "$OUT/$name.manifest.json" 2>&1 | tail -16
}

# ---- training ----
run_bpseq bprna_tr0    "$B/bpRNA/TR0"
run_bpseq bprna_vl0    "$B/bpRNA/VL0"
run_bpseq rnastralign  "$B/RNAStrAlign_bpseq"
run_bpseq pdb669       "$B/PDB_669/PDB"

# ---- in-distribution test ----
run_bpseq bprna_ts0    "$B/bpRNA/TS0"
run_csv   archiveii    "$RINALMO/ArchiveII.csv"

# ---- cross-family OOD ----
run_bpseq bprna_new    "$X/bpRNAnew/bpRNAnew.nr500.canonicals"
run_bpseq rfam_fam     "$B/Rfam12.3-14.10"

# ---- experimental labels ----
run_bpseq pdb_ts1      "$X/PDB_test/TS1"
run_bpseq pdb_ts2      "$X/PDB_test/TS2"
run_bpseq pdb_ts3      "$X/PDB_test/TS3"

echo "### ALL DONE  ($(date +%H:%M:%S))"
