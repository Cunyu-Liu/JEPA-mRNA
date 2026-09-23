#!/bin/bash
# Acquisition orchestrator for the RNA secondary-structure datasets (spec §3.3).
#
# There is NO network access on this machine, and large artefacts live on the
# cluster at /mnt/cunyuliu/rna-jepa.  This script therefore defaults to a
# DRY-RUN: it prints exactly what would be fetched (URL, size, md5, the sharded
# download command) and never touches the network.
#
# The actual byte transfer is delegated to the proven multi-connection ranged
# downloader scripts/fetch_zenodo_parallel.sh (DNS-poisoned host + ~15-20 KB/s
# per connection => fixed-size byte ranges, many workers, .parts/p%06d shards
# for resumability, then concat + md5).  We reuse it rather than reimplement it.
#
# Usage:
#   scripts/fetch_ss_data.sh                     # dry-run plan (default)
#   scripts/fetch_ss_data.sh --only bprna_1m     # dry-run for one source
#   scripts/fetch_ss_data.sh --execute           # real fetch (needs network)
#
# NOTE: several sources have unknown URLs/sizes and are marked 待核验; the plan
# reports them as BLOCKED.  They cannot be fetched until their URL/size is
# recorded -- no URL or hash is ever invented.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"

OUT_ROOT="${RNAJEPA_SS_ROOT:-/mnt/cunyuliu/rna-jepa/data/ss/raw}"
MANIFEST="${RNAJEPA_SS_MANIFEST:-$ROOT/data/ss/hash_manifest.json}"

echo "=== RNA SS data acquisition (spec §3.3) ==="
echo "out_root = $OUT_ROOT"
echo "manifest = $MANIFEST"
echo "helper   = $ROOT/scripts/fetch_zenodo_parallel.sh"
echo

if [[ "${1:-}" == "--execute" ]]; then
  shift
  echo "[mode] EXECUTE (requires network; blocked specs are skipped)"
  exec "$PY" "$ROOT/data/ss/fetch.py" --execute --out-root "$OUT_ROOT" --manifest "$MANIFEST" "$@"
else
  echo "[mode] DRY-RUN (no network access)"
  exec "$PY" "$ROOT/data/ss/fetch.py" --dry-run --out-root "$OUT_ROOT" --manifest "$MANIFEST" "$@"
fi
