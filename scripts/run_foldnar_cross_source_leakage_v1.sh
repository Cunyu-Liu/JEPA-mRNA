#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/rna_jev_cross_source_leakage}"
V2_REPO="${V2_REPO:-/home/rna_jev_model_dev_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/teams/HoulabData_1788502323/liucunyu/foldnar/cross_source_leakage_v1}"
TMP_ROOT="${TMP_ROOT:-/tmp/foldnar_cross_source_leakage_v1_mmseqs_tmp}"
PYTHON="${PYTHON:-/root/miniconda3/envs/lucaone_cu128/bin/python}"
MMSEQS="${MMSEQS:-/home/rna_jev/tools/bin/mmseqs}"
THREADS="${THREADS:-2}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export CUDA_VISIBLE_DEVICES=""

nice -n 10 "$PYTHON" "$REPO/tools/audit_foldnar_cross_source_leakage_v1.py" \
  --repo "$REPO" \
  --v2-repo "$V2_REPO" \
  --v2-registry "$V2_REPO/manifests/foldnar_v2/model_dev_registry.json" \
  --mmseqs "$MMSEQS" \
  --output-root "$OUTPUT_ROOT" \
  --tmp-root "$TMP_ROOT" \
  --threads "$THREADS"

nice -n 10 "$PYTHON" "$REPO/tools/verify_foldnar_cross_source_leakage_v1.py" \
  --artifact-root "$OUTPUT_ROOT"
