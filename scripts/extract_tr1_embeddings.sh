#!/bin/bash
# Extract RiNALMo-giga embeddings for the expanded corpus TR1.
#
# Why this is a separate step
# ---------------------------
# The frozen-embedding arms (`rinalmo_ff`, `rinalmo_len`, ...) do not run RiNALMo
# during training; they read precomputed shards from
# `/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga/<split>.shardKofN.npz`.  Those
# shards existed for TR0 and the eval splits but not for TR1, and the trainer
# correctly refused to start:
#
#   FATAL: ConfigError: embedding shards under .../rinalmo-giga contained no
#   sequences matching split='bprna_tr1'
#
# That refusal is the desired behaviour (a silent fallback to random embeddings
# would have produced a plausible-looking but meaningless run), so this step has
# to exist rather than be worked around.
#
# Concurrency and the manifest
# ----------------------------
# `tools/extract_rinalmo_embeddings.py` reads the manifest, merges its own entry,
# and rewrites it.  Two concurrent shards would therefore each drop the other's
# entry -- a lost update, not a crash, which is the worse failure mode.  Each
# shard here writes to its own manifest file, and one merge at the end unions
# them.  The shard .npz files themselves are already unique per shard
# (`<split>.shardKofN.npz`), so only the manifest needed care.
#
# Usage: bash scripts/extract_tr1_embeddings.sh [n_shards] [batch_size]
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
OUT="${RNAJEPA_EMB:-/mnt/cunyuliu/rna-jepa/embeddings/rinalmo-giga}"
PY="${RNAJEV_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"
SPLIT="${RNAJEV_SPLIT:-bprna_tr1}"
N="${1:-2}"
BS="${2:-2}"

export PYTHONPATH=/mnt/cunyuliu/pylibs:src
export TMPDIR=/mnt/cunyuliu/tmp
cd "$ROOT"

# Devices, in launch order.  Overridable because the right choice is a
# *measurement*, not a constant: on 2026-09-24 18:20 the two 3g.20gb instances on
# GPU 7 had ~5.9 GiB free each, and at 18:40 the picture had changed again
# (GPU 0 13.6 GiB, GPU 6 14.0 GiB, GPU 7 15.9 GiB free) as other users' jobs and
# our own TR0 arms finished.  Default is the MIG UUIDs, which are exact; set
# RNAJEV_DEVS="0 1" to use whole-GPU indices instead.
if [ -n "${RNAJEV_DEVS:-}" ]; then
  read -r -a DEVS <<< "$RNAJEV_DEVS"
else
  DEVS=("MIG-6e59f9af-2716-5bf2-ac6e-fb05ef744585"
        "MIG-10b9b777-a776-56de-9f7c-efde6c584f71")
fi

echo "=== embedding extraction: split=$SPLIT shards=$N batch=$BS ($(date +%H:%M:%S)) ==="
for k in $(seq 0 $((N - 1))); do
  dev="${DEVS[$((k % ${#DEVS[@]}))]}"
  mf="$OUT/.manifest_${SPLIT}_shard${k}.json"
  log="$OUT/.extract_${SPLIT}_shard${k}.log"
  # idempotency: an existing shard file is reused rather than recomputed
  if ls "$OUT/${SPLIT}.shard${k}of${N}.npz" >/dev/null 2>&1; then
    echo "[skip] shard $k already on disk"
    continue
  fi
  echo "[launch] shard $k -> $dev"
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" tools/extract_rinalmo_embeddings.py \
      --split "$SPLIT" --shards "$N" --shard "$k" \
      --batch-size "$BS" --device cuda --manifest "$mf" \
      > "$log" 2>&1 < /dev/null &
  echo "[launch] shard $k pid=$!"
done

# Grace period before the first liveness check.  Importing torch takes several
# seconds, so checking immediately can see zero matching processes and break out
# of the wait loop *before the workers have even started* -- which then makes the
# merge step report missing shards and exit 3, i.e. a spurious hard failure that
# looks exactly like a real one.
startup_grace=25
sleep "$startup_grace"

wait_seconds=$startup_grace
while :; do
  alive=$(ps -eo args | grep -c "[e]xtract_rinalmo_embeddings.py" || true)
  [ "$alive" -eq 0 ] && break
  sleep 20
  wait_seconds=$((wait_seconds + 20))
  [ "$wait_seconds" -gt 5400 ] && { echo "TIMEOUT after 90 min"; break; }
done
echo "=== workers finished ($(date +%H:%M:%S)) ==="

"$PY" - "$OUT" "$SPLIT" "$N" <<'PYEOF'
"""Merge the per-shard manifests into the canonical one, without losing entries.

Reads every ``.manifest_<split>_shard*.json`` plus whatever is already in
``manifest.json``, unions them by (split, shard), and rewrites.  Reports which
shards are present so a partial extraction is visible rather than implied.
"""
import json, os, sys

out, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
entries = {}
canonical = os.path.join(out, "manifest.json")
if os.path.isfile(canonical):
    with open(canonical, encoding="utf-8") as fh:
        for e in json.load(fh).get("entries", []):
            entries[(e["split"], e["shard"])] = e
for k in range(n):
    p = os.path.join(out, f".manifest_{split}_shard{k}.json")
    if not os.path.isfile(p):
        continue
    with open(p, encoding="utf-8") as fh:
        for e in json.load(fh).get("entries", []):
            entries[(e["split"], e["shard"])] = e
payload = {"model": "multimolecule/rinalmo-giga", "dtype": "float16",
           "license_note": "AGPL-3.0; confirm compliance before publication",
           "tokenizer": "hand-built from vocab.txt (no AutoTokenizer)",
           "entries": sorted(entries.values(), key=lambda e: (e["split"], e["shard"]))}
with open(canonical, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=1, ensure_ascii=False)
have = sorted(k for (s, k) in entries if s == split)
print(f"[manifest] {canonical}: {len(entries)} entries; "
      f"split={split} shards present={have} (expected {n})")
missing = [k for k in range(n) if k not in have]
if missing:
    print(f"FATAL: shards missing from the manifest: {missing}", file=sys.stderr)
    sys.exit(3)
PYEOF
rc=$?
echo "### extraction done rc=$rc ($(date +%H:%M:%S))"
exit $rc