#!/bin/bash
# Build the JSONL structure corpora for the decision model.
#
# Why this script audits *contents* and not just file counts
# ----------------------------------------------------------
# The first version of this inventory counted files.  That was wrong, and it cost
# real time: /mnt/cunyuliu/BPfold_data is a truncated 2025 extraction in which
# RNAStrAlign_bpseq (37,052 files) and Rfam12.3-14.10 (10,791 files) are *all
# zero bytes*, and archiveII/archiveII is an empty directory.  Counting names
# reported all three as present and healthy.  The corpus build then produced a
# 0-byte rnastralign.jsonl with 1 MB of "empty: no rows" rejects, which is how
# the problem surfaced.
#
# Every input is therefore checked for usable (non-zero) content before anything
# is built, the check is recorded in the manifest, and a required input with zero
# usable files is a hard failure rather than a silent skip.
#
# Where the data actually is
# --------------------------
# The test splits live in the BPfold release tarball
#   /mnt/cunyuliu/rna_ss_data/bpfold/BPfold_test_results/BPfold_test_results/
# which carries archiveII (3,966), Rfam12.3-14.10 (10,791), Rfam14.10-15.0 (436),
# bpRNAnew (5,401), bpRNA (1,305) and PDB_test (116) -- all verified non-empty.
# Training splits come from /mnt/cunyuliu/BPfold_data (bpRNA/TR0 10,814,
# PDB_669 669), which are also verified non-empty.
#
# Policy (see spec/benchmark_decision.md): --min-loop-policy drop
# --pseudoknot-policy drop.  The decision head and harness can only represent
# nested structures with hairpin loops >= 3, so the ground truth has to be
# projected into that space for F1 to mean anything.  Rejecting instead would
# discard 4.6% of TR0 and 71% of PDB ts3, and ts3 is part of the only
# experimental-label set we have.
#
# Usage: bash scripts/build_ss_corpus.sh [output_dir]
set -uo pipefail

PY="${RNAJEV_PYTHON:-$HOME/miniconda3/envs/lucaone/bin/python}"
OUT="${1:-/mnt/cunyuliu/rna-jepa/ss_data/jsonl}"

B=/mnt/cunyuliu/BPfold_data
X=/mnt/cunyuliu/rna_ss_data/bpfold/extracted/BPfold_data
T=/mnt/cunyuliu/rna_ss_data/bpfold/BPfold_test_results/BPfold_test_results
RINALMO=/mnt/cunyuliu/rna_ss_data/rinalmo

mkdir -p "$OUT"

# ---- 0. content audit -----------------------------------------------------
audit_dir() {  # audit_dir <dir> -> "<usable> <zero>"
  [ -d "$1" ] || { echo "0 0"; return; }
  find "$1" -maxdepth 1 -type f -name "*.bpseq" -printf "%s\n" 2>/dev/null \
    | awk '{if ($1 == 0) z++; else u++} END {printf "%d %d", u+0, z+0}'
}

AUDIT_LOG="$OUT/_content_audit.txt"
: > "$AUDIT_LOG"
FAILED=0
check() {  # check <label> <dir> <required:1|0>
  local label="$1" dir="$2" required="$3"
  read -r usable zero <<<"$(audit_dir "$dir")"
  printf '%-22s usable=%-7d zero=%-7d %s\n' "$label" "$usable" "$zero" "$dir" \
    | tee -a "$AUDIT_LOG"
  if [ "$usable" -eq 0 ] && [ "$required" -eq 1 ]; then
    echo "FATAL: required input '$label' has zero usable files at $dir" >&2
    FAILED=1
  fi
}

echo "=== content audit ($(date +%H:%M:%S)) ==="
check bpRNA_TR0       "$B/bpRNA/TR0"        1
check bpRNA_VL0       "$B/bpRNA/VL0"        1
check bpRNA_TS0       "$B/bpRNA/TS0"        1
check PDB_669         "$B/PDB_669/PDB"      1
check archiveII       "$T/archiveII"        1
check Rfam12.3-14.10  "$T/Rfam12.3-14.10"   1
check Rfam14.10-15.0  "$T/Rfam14.10-15.0"   0
check bpRNAnew        "$T/bpRNAnew"         1
check PDB_TS1         "$T/PDB_test"         1
check RNAStrAlign     "$B/RNAStrAlign_bpseq" 0
[ "$FAILED" -eq 1 ] && exit 2

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
run_bpseq pdb669       "$B/PDB_669/PDB"

# ---- in-distribution test ----
run_bpseq bprna_ts0    "$B/bpRNA/TS0"
run_bpseq archiveii    "$T/archiveII"
run_csv   archiveii_rinalmo "$RINALMO/ArchiveII.csv"

# ---- cross-family / temporal OOD ----
run_bpseq bprna_new    "$T/bpRNAnew"
run_bpseq rfam_fam     "$T/Rfam12.3-14.10"
run_bpseq rfam_temporal "$T/Rfam14.10-15.0"

# ---- experimental labels ----
run_bpseq pdb_ts_all   "$T/PDB_test"

echo "### ALL DONE  ($(date +%H:%M:%S))"
