#!/bin/bash
# Wait for the TR1 embeddings to land, then start TR1 training.
#
# Why a chain rather than two manual steps
# ----------------------------------------
# TR1 training cannot start until `bprna_tr1` embedding shards exist -- the
# trainer refuses with
#
#   FATAL: ConfigError: embedding shards ... contained no sequences matching
#   split='bprna_tr1'
#
# and that refusal is correct.  But the embedding step finishes at an
# unpredictable time (the final `np.savez` of a ~6 GB object array dominates and
# is not proportional to sequence count), so the launch has to be event-driven
# rather than scheduled by a guessed delay.
#
# The gate is the *manifest*, not the process list: a shard file can exist while
# still being written, whereas the manifest entry is written after the file.
# Both conditions are checked, and the merged manifest is re-read so that the
# normalisation step from `scripts/extract_tr1_embeddings.sh` has run.
#
# Usage: setsid nohup bash scripts/await_tr1_embeddings_then_launch.sh > ... 2>&1 < /dev/null
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
PY="${RNAJEV_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"
E="$ART/embeddings/rinalmo-giga"
SPLIT="${RNAJEV_SPLIT:-bprna_tr1}"
N="${RNAJEV_SHARDS:-2}"
DEADLINE=$(( $(date +%s) + 7200 ))   # give up after 2 h rather than hang forever

echo "=== awaiting embeddings for $SPLIT (shards=$N) ($(date +%H:%M:%S)) ==="
while :; do
  ready=1
  for k in $(seq 0 $((N - 1))); do
    [ -s "$E/${SPLIT}.shard${k}of${N}.npz" ] || ready=0
  done
  if [ "$ready" -eq 1 ] && "$PY" - "$E" "$SPLIT" "$N" <<'PYEOF'
import json, os, sys
out, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = os.path.join(out, "manifest.json")
if not os.path.isfile(p):
    sys.exit(1)
entries = {(e["split"], e["shard"]) for e in json.load(open(p)).get("entries", [])}
sys.exit(0 if all((split, k) in entries for k in range(n)) else 1)
PYEOF
  then
    echo "[ready] all $N shards present and in the manifest ($(date +%H:%M:%S))"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "TIMEOUT waiting for embeddings after 2 h" >&2
    exit 3
  fi
  sleep 30
done

cd "$ROOT"
echo "[launch] starting TR1 training ($(date +%H:%M:%S))"
bash scripts/launch_tr1.sh
echo "### chain complete ($(date +%H:%M:%S))"