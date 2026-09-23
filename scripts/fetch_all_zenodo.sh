#!/bin/bash
# Launch all Zenodo fetches for RNA-JEPA. Run inside tmux/nohup.
#
# Ordering rationale: the 247 MB downstream bundle unblocks the R3 baseline
# gate (critical path), so it goes first with aggressive parallelism. The
# 20 GB pre-training archive is the long pole and starts right after.
#
# Per-connection throughput to Zenodo is throttled to ~15-20 KB/s from this
# cluster, but aggregate scales ~linearly with connection count, so we use
# small chunks (more chunks = more concurrent connections).
set -uo pipefail
S=/home/cunyuliu/rna-jepa/scripts/fetch_zenodo_parallel.sh
RAW=/mnt/cunyuliu/rna-jepa/data/raw
mkdir -p "$RAW"
API=https://zenodo.org/api/records

fetch_bg() { # url out size md5 jobs chunk_mb
  bash $S "$1" "$2" "$3" "$4" "$5" "$6" > "${2}.fetch.log" 2>&1 &
  echo $! > "${2}.pid"
}

echo "=== stage 1: downstream (17786045) ==="
fetch_bg "$API/17786045/files/full_length.zip/content"          "$RAW/full_length.zip"          281158    3652178c257341010800e2d241a9c258 4   1
fetch_bg "$API/17786045/files/Spliceator.zip/content"           "$RAW/Spliceator.zip"           24191813  d80c393eec09728c57e2c66c361e90e9 12  2
fetch_bg "$API/17786045/files/protein.zip/content"              "$RAW/protein.zip"              30702614  e8f6b277303959a2010d0047830b2c83 16  2
fetch_bg "$API/17786045/files/te_ultra_full_length.zip/content" "$RAW/te_ultra_full_length.zip"  32611876  939b495793687db362d4b9464a5df570 16  2
fetch_bg "$API/17786045/files/CDS.zip/content"                  "$RAW/CDS.zip"                  35586644  dbb145edd36a67b63e7184da04dab8c4 18  2
fetch_bg "$API/17786045/files/5UTR.zip/content"                 "$RAW/5UTR.zip"                 60237546  6d36a52b06b6d493e900d60590e881da 30  2
fetch_bg "$API/17786045/files/3UTR.zip/content"                 "$RAW/3UTR.zip"                 64124658  6edf8dffcb2ee63560e276a65a5f5e9f 32  2
wait
echo "DOWNSTREAM_ALL_DONE"

echo "=== stage 2: pretrain (12516160, 20 GB) ==="
fetch_bg "$API/12516160/files/mRNAdataset.zip/content" "$RAW/mRNAdataset.zip" 20021485711 bf8bc5c946a0bd3b07716b1c7f785d54 168 8
wait
echo "PRETRAIN_DONE"