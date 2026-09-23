#!/bin/bash
# Baseline reproduction runner for RNA secondary-structure prediction (spec §7.2).
#
# Covers every route in §7.2.2 plus the §7.2.1 closest prior work.  No baseline
# tool is installed in this environment and there is no network, so this script
# defaults to DRY-RUN: it prints the full matrix and a per-baseline plan
# (version lock + install hint), and enforces the G5 provenance gate.
#
# It NEVER fabricates baseline outputs.  A real run requires the tool to be
# installed and its commit + weight hash to be recorded first; until then
# eval/ss/baselines.py raises an actionable ToolUnavailableError.
#
# Usage:
#   scripts/run_baselines.sh                 # matrix + dry-run plans + G5 gate
#   scripts/run_baselines.sh --key ufold     # plan for one baseline
#   scripts/run_baselines.sh --run           # attempt real runs (fails until installed)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"
B="$ROOT/eval/ss/baselines.py"

echo "=== RNA SS baseline matrix (spec §7.2) ==="
"$PY" "$B" --matrix || true
echo

if [[ "${1:-}" == "--run" ]]; then
  shift
  echo "[mode] RUN (attempt real baselines; ToolUnavailableError expected until installed)"
  KEYS="$("$PY" - "$ROOT" <<'PYEOF'
import sys, os
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "eval", "ss"))
sys.path.insert(0, os.path.join(root, "src"))
import baselines
print("\n".join(b.key for b in baselines.registry()))
PYEOF
)"
  for k in $KEYS; do
    echo "--- $k ---"
    "$PY" "$B" --key "$k" "$@" || echo "[baselines] $k unavailable (see error above)"
  done
else
  KEY="${2:-}"
  if [[ -n "$KEY" ]]; then
    echo "[mode] DRY-RUN plan for $KEY"
    "$PY" "$B" --dry-run --key "$KEY"
  else
    echo "[mode] DRY-RUN plans for all baselines"
    "$PY" - "$ROOT" <<'PYEOF'
import json, sys, os
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "eval", "ss"))
sys.path.insert(0, os.path.join(root, "src"))
import baselines
for b in baselines.registry():
    print(json.dumps(baselines.plan_baseline(b.key), ensure_ascii=False))
PYEOF
  fi
  echo
  echo "[note] closest prior work (must not be dropped): CONTRAfold, CDPFold, LinearPartition, E2Efold"
  echo "[note] LinearPartition speedup is reported separately (never cherry-pick the opponent)"
fi
