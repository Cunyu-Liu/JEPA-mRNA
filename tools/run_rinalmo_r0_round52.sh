#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_r0.lock
flock -n 9 || { echo "another R0 queue is active"; exit 1; }
"$PY" - "$ROOT/audits/round52/rinalmo_r0_smoke_qa.json" <<'PY'
import json
import sys
qa = json.load(open(sys.argv[1]))
assert qa["status"] == "PASS_R0_SMOKE_THREE_CHECK_QA"
assert not qa["scientific_conclusion"]
PY
train_seed() {
  local seed=$1
  local out="$ROOT/runs/round52_rinalmo_r0_s${seed}"
  mkdir -p "$out"
  if [[ -f "$out/run_meta.json" ]] && "$PY" - "$out" "$seed" <<'PY'
import json
import sys
from pathlib import Path
root, seed = Path(sys.argv[1]), int(sys.argv[2])
meta = json.loads((root / "run_meta.json").read_text())
assert meta["arm"] == f"round52_rinalmo_r0_s{seed}" and meta["seed"] == seed
assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
assert meta["schedule"]["steps"] == 0 and meta["schedule"]["batch_size"] == 2
assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
assert meta["objective"]["nll_normalization"] == "sum"
assert meta["scorer"] == "pairmlp" and meta["pair_latent_dim"] == 490
assert meta["ripe"]["mode"] == "r6"
assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
PY
  then
    echo "skip converged R0 seed $seed"
    return 0
  fi
  "$PY" "$REPO/tools/train_plan_a.py" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round52" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
    --zchunk 64 --gpu-reserve-gb 3 --scorer pairmlp --pair-latent-dim 490 --ripe-mode r6 \
    >> "$ROOT/logs/round52/rinalmo_r0_s${seed}.log" 2>&1
}
for seed in 42 2026 3407; do
  train_seed "$seed"
done
"$PY" - "$ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
runs = {}
for seed in (42, 2026, 3407):
    run = root / "runs" / f"round52_rinalmo_r0_s{seed}"
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
    assert meta["stop_reason"] == "early_stopping" and meta["schedule"]["steps"] == 0
    assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
    runs[str(seed)] = {
        "best_step": meta["best_step"], "best_vl0": meta["best_validation_loss"],
        "steps_completed": meta["steps_completed"],
        "best_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
        "resume_sha256": hashlib.sha256((run / "resume.pt").read_bytes()).hexdigest(),
    }
out = root / "audits/round52/rinalmo_r0_training_complete.json"
out.write_text(json.dumps({"status": "PASS_R0_THREE_SEED_NATURAL_CONVERGENCE",
                           "scientific_conclusion": False, "runs": runs}, indent=2) + "\n")
print(json.dumps({"status": "R0_TRAINING_COMPLETE", "runs": runs}, indent=2))
PY
