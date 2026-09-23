#!/bin/bash
# Push the local working copy to the cluster code tree.
#
# The cluster is the execution authority (/home/rna-jepa); this Mac copy is the
# editing surface.  Large artefacts never travel this path: they live under
# /mnt/cunyuliu on the cluster and are referenced by path.
#
# Usage: scripts/sync_to_cluster.sh [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${RNAJEPA_REMOTE:-A100}"
REMOTE_DIR="${RNAJEPA_REMOTE_DIR:-/home/cunyuliu/rna-jepa}"

EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude 'third_party/mRNABERT/data_process/pre-train/pre_input.fasta'
  --exclude 'weights_ref/mRNABERT/pytorch_model.bin'
  --exclude '.DS_Store'
)

echo "[sync] $HERE -> $REMOTE:$REMOTE_DIR"
rsync -az --delete "${EXCLUDES[@]}" "$@" "$HERE/" "$REMOTE:$REMOTE_DIR/"
echo "[sync] done"