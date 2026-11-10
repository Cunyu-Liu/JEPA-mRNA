#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_capacity_diagnostics.lock
flock -n 9 || { echo "another capacity diagnostic queue is active"; exit 1; }
for arm in length olr; do
  for seed in 42 2026 3407; do
    run="$ROOT/runs/round52_rinalmo_${arm}_s${seed}"
    until "$PY" - "$run" "$arm" "$seed" <<'PY'
import json
import sys
from pathlib import Path
run, arm, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path = run / "run_meta.json"
if not path.exists():
    raise SystemExit(1)
meta = json.loads(path.read_text())
assert meta["arm"] == f"round52_rinalmo_{arm}_s{seed}" and meta["seed"] == seed
assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
assert meta["schedule"]["steps"] == 0
assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
assert meta["environment"]["resolved_device"].startswith("cuda")
PY
    do
      sleep 300
    done
  done
done
"$PY" "$REPO/tools/qa_rinalmo_loss_optimizer_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_loss_optimizer_training_qa.json"
"$PY" "$REPO/tools/qa_rinalmo_loss_optimizer_round52.py" --root "$ROOT" \
  --output "$ROOT/audits/round52/rinalmo_loss_optimizer_training_qa.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_loss_optimizer_training_qa.json" \
    "$ROOT/audits/round52/rinalmo_loss_optimizer_training_qa.recheck.json"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
while (( used > 14000 )); do
  sleep 300
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
done
for arm in length olr; do
  for seed in 42 2026 3407; do
    checkpoint="$ROOT/runs/round52_rinalmo_${arm}_s${seed}/best.pt"
    error="$ROOT/audits/round52/rinalmo_capacity_error_${arm}_s${seed}.json"
    error_qa="$ROOT/audits/round52/rinalmo_capacity_error_${arm}_s${seed}_qa.json"
    representation="$ROOT/audits/round52/rinalmo_capacity_representation_${arm}_s${seed}.json"
    representation_qa="$ROOT/audits/round52/rinalmo_capacity_representation_${arm}_s${seed}_qa.json"
    if [[ -e "$error" || -e "$error_qa" || -e "$representation" || -e "$representation_qa" ]]; then
      echo "refusing to overwrite unreviewed capacity artifacts for $arm $seed" >&2
      exit 1
    fi
    "$PY" "$REPO/tools/probe_rinalmo_capacity_error_round52.py" \
      --checkpoint "$checkpoint" \
      --train-data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
      --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
      --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
      --validation-teacher-dir "$ROOT/data/teacher_vl0" \
      --train-examples 196 --seed 42 --output "$error" \
      > "$ROOT/logs/round52/rinalmo_capacity_error_${arm}_s${seed}.log" 2>&1
    "$PY" "$REPO/tools/qa_rinalmo_capacity_error_round52.py" \
      --input "$error" --probe-code "$REPO/tools/probe_rinalmo_capacity_error_round52.py" \
      --output "$error_qa" > "$ROOT/logs/round52/rinalmo_capacity_error_${arm}_s${seed}_qa.log" 2>&1
    "$PY" "$REPO/tools/probe_rinalmo_frozen_representation_round52.py" \
      --checkpoint "$checkpoint" \
      --train-data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
      --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
      --samples-per-sequence 256 --seed 42 --output "$representation" \
      > "$ROOT/logs/round52/rinalmo_capacity_representation_${arm}_s${seed}.log" 2>&1
    "$PY" "$REPO/tools/qa_rinalmo_frozen_representation_round52.py" \
      --input "$representation" --probe-code "$REPO/tools/probe_rinalmo_frozen_representation_round52.py" \
      --output "$representation_qa" \
      > "$ROOT/logs/round52/rinalmo_capacity_representation_${arm}_s${seed}_qa.log" 2>&1
  done
done
"$PY" - "$ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
files = {}
for arm in ("length", "olr"):
    for seed in (42, 2026, 3407):
        for kind in ("error", "representation"):
            path = root / "audits/round52" / f"rinalmo_capacity_{kind}_{arm}_s{seed}_qa.json"
            report = json.loads(path.read_text())
            assert not report["allow_capacity_expansion"] and not report["scientific_conclusion"]
            files[f"{kind}_{arm}_s{seed}"] = hashlib.sha256(path.read_bytes()).hexdigest()
out = root / "audits/round52/rinalmo_capacity_diagnostics_complete.json"
out.write_text(json.dumps({
    "status": "PASS_ALL_L_O_ERROR_AND_REPRESENTATION_PROBES",
    "scientific_conclusion": False, "allow_capacity_expansion": False,
    "allow_j_gate": False, "qa_sha256": files,
    "limitations": ["All six runs and both diagnostic families are complete; capacity decision and independent replication remain required.",
                    "No test set was accessed and no C-wide or C-local run is licensed yet."]}, indent=2) + "\n")
print(json.dumps({"status": "CAPACITY_DIAGNOSTICS_COMPLETE", "qa_sha256": files}, indent=2))
PY
