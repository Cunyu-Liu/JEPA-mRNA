#!/usr/bin/env bash
set -euo pipefail

pid="$1"
run_dir="$2"

while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

/root/miniconda3/envs/lucaone/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]) / "run.json"
if not run.is_file():
    raise SystemExit(f"active training ended without run.json; refusing automatic retry: {run}")
result = json.loads(run.read_text())
required = {"status": "PASS", "converged": True, "early_stopped": True, "scientific_eligibility": True}
for key, expected in required.items():
    if result.get(key) != expected:
        raise SystemExit(f"training completion gate failed: {key}={result.get(key)!r}, expected={expected!r}")
print(f"COMPLETION_GATE_PASS {run}", flush=True)
PY

cd /home/rna_jev
export PYTHONPATH=/home/rna_jev/src:/home/rna_jev/tools
export NB_CKPT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt
exec flock -n /tmp/round35_nb_b4.lock /root/miniconda3/envs/lucaone/bin/python \
  tools/run_nb_secstr_queue_round35.py --parallel 1 --min-free-mib 8000 --interval 30
