#!/usr/bin/env bash
set -euo pipefail

cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
PYTHON=/root/miniconda3/envs/lucaone/bin/python
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_xens2
DATA=/root/rna_jepa_data/ss_data/jsonl
EMB=/root/rna_jepa_data/embeddings/rinalmo-giga
INIT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/main_ablations/r2d.pt
mkdir -p logs/round39_xens2 "$ROOT/runs"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
from rnajepa.distill import verify_teacher_labels
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_xens2")
for name, expected in (("teacher_tr1c", 42564), ("teacher_vl0", 196)):
    path = root / name
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["n_sequences"] == expected
    assert manifest["meta"]["teacher_batch_size"] == 1
    assert manifest["meta"]["allow_tf32"] is False
    assert verify_teacher_labels(manifest, path)
    print("TEACHER_VERIFIED", name, expected, len(manifest["shards"]), flush=True)
PY

common=(
  --data "$DATA/bprna_tr1c.jsonl"
  --teacher-dir "$ROOT/teacher_tr1c"
  --validation-data "$DATA/bprna_vl0.jsonl"
  --validation-teacher-dir "$ROOT/teacher_vl0"
  --embedding-dir "$EMB"
  --validation-embedding-dir "$EMB"
  --embedding-d-model 1280
  --head-chunk-size 0
  --scorer resnet2d
  --resnet-blocks 4
  --d-z 128
  --hidden 64
  --init-weights "$INIT"
  --steps 1000000
  --batch-size 2
  --lr 1e-4
  --warmup-steps 5
  --lambda-nll 1
  --lambda-distill 1
  --lambda-rlcd 1
  --lambda-cal 1
  --nll-normalization sum
  --validation-every 500
  --validation-patience 5
  --validation-min-delta 1e-6
  --log-every 25
  --save-every 500
  --device cuda
)

"$PYTHON" src/rnajepa/train_decision.py "${common[@]}" \
  --seed 0 --arm xens2_distill_r2d_s0 \
  --out "$ROOT/runs/xens2_distill_r2d_s0" \
  --dry-run > "$ROOT/distill_dry_run.json"
"$PYTHON" - <<'PY'
import json
p = "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_xens2/distill_dry_run.json"
d = json.load(open(p))
assert d["dry_run"] and d["steps"] == 1000000 and d["batch_size"] == 2 and d["device"] == "cuda"
assert d["data"]["n_examples"] == 42564
v = d["validation"]
assert v["data"]["n_examples"] == 196
assert v["early_stopping"] and v["steps_are_safety_cap"]
assert v["every"] == 500 and v["patience"] == 5 and v["min_delta"] == 1e-6
assert d["objective"] == {"nll": 1.0, "distill": 1.0, "rlcd": 1.0, "cal": 1.0}
print("DISTILL_DRY_RUN_PASS", d["steps_per_epoch"], d["epochs"], flush=True)
PY

pids=()
read -r -a seeds <<< "${SEEDS:-0 1 2}"
for seed in "${seeds[@]}"; do
  arm="xens2_distill_r2d_s${seed}"
  out="$ROOT/runs/$arm"
  test ! -e "$out/run_meta.json"
  "$PYTHON" src/rnajepa/train_decision.py "${common[@]}" \
    --seed "$seed" --arm "$arm" --out "$out" \
    > "logs/round39_xens2/${arm}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  printf '%s %s %s\n' "$seed" "$arm" "$pid" | tee -a "$ROOT/training_pids.txt"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "DISTILL_TRAINING_FAILED; inspect per-seed logs before any resume" >&2
  exit "$status"
fi

SEEDS="${SEEDS:-0 1 2}" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_xens2/runs")
for seed in map(int, os.environ["SEEDS"].split()):
    path = root / f"xens2_distill_r2d_s{seed}"
    meta = json.loads((path / "run_meta.json").read_text())
    assert meta["converged"] and meta["early_stopped"], (seed, meta.get("stop_reason"))
    assert (path / "best.pt").is_file() and (path / "resume.pt").is_file()
    assert meta["environment"]["resolved_device"].startswith("cuda")
    print("DISTILL_CONVERGED", seed, meta["steps_completed"], meta["best_step"], flush=True)
print("DISTILL_SEEDS_READY_FOR_INDEPENDENT_GPU_QA", os.environ["SEEDS"])
PY
