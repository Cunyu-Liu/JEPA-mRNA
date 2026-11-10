#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_loss_optimizer.lock
flock -n 9 || { echo "another L/O queue is active"; exit 1; }
"$PY" - "$ROOT/audits/round52/rinalmo_replay_decision.json" <<'PY'
import json
import sys
qa = json.load(open(sys.argv[1]))
assert qa["status"] == "PASS_R_REPLAY_THREE_CHECK_QA"
assert qa["allow_l_length_single_factor"] and qa["allow_o_lr_single_factor"]
assert not qa["allow_loss_optimizer_interaction"] and not qa["allow_capacity_stage"]
PY
train_one() {
  local arm=$1 seed=$2 normalization backbone_lr out
  out="$ROOT/runs/round52_rinalmo_${arm}_s${seed}"
  if [[ "$arm" == length ]]; then normalization=length; backbone_lr=1e-5;
  else normalization=sum; backbone_lr=1e-6; fi
  mkdir -p "$out"
  if [[ -f "$out/run_meta.json" ]] && "$PY" - "$out" "$arm" "$seed" "$normalization" "$backbone_lr" <<'PY'
import json
import sys
from pathlib import Path
root, arm, seed, normalization, lr = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
m = json.loads((root / "run_meta.json").read_text())
assert m["arm"] == f"round52_rinalmo_{arm}_s{seed}" and m["seed"] == seed
assert m["status"] == "completed" and m["converged"] and m["early_stopped"]
assert m["schedule"]["steps"] == 0 and m["schedule"]["batch_size"] == 2
assert m["schedule"]["backbone_lr"] == lr
assert m["validation"]["every"] == 500 and m["validation"]["patience"] == 6
assert m["objective"]["nll_normalization"] == normalization
assert m["scorer"] == "pairmlp" and m["pair_latent_dim"] == 490
assert m["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
PY
  then
    echo "skip converged $arm $seed"
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
    --head-lr 1e-4 --backbone-lr "$backbone_lr" --nll-normalization "$normalization" \
    --zchunk 64 --gpu-reserve-gb 3 --scorer pairmlp --pair-latent-dim 490 --ripe-mode r6 \
    >> "$ROOT/logs/round52/rinalmo_${arm}_s${seed}.log" 2>&1
}
for job in length:42 olr:42 length:2026 olr:2026 length:3407 olr:3407; do
  train_one "${job%%:*}" "${job##*:}"
done
"$PY" - "$ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
runs = {}
for arm in ("length", "olr"):
    for seed in (42, 2026, 3407):
        path = root / "runs" / f"round52_rinalmo_{arm}_s{seed}"
        meta = json.loads((path / "run_meta.json").read_text())
        assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
        assert meta["schedule"]["steps"] == 0
        runs[f"{arm}_s{seed}"] = {
            "best_step": meta["best_step"], "best_vl0": meta["best_validation_loss"],
            "steps_completed": meta["steps_completed"],
            "best_sha256": hashlib.sha256((path / "best.pt").read_bytes()).hexdigest(),
            "resume_sha256": hashlib.sha256((path / "resume.pt").read_bytes()).hexdigest(),
        }
out = root / "audits/round52/rinalmo_loss_optimizer_training_complete.json"
out.write_text(json.dumps({"status": "PASS_L_O_THREE_SEED_NATURAL_CONVERGENCE",
                           "scientific_conclusion": False, "runs": runs}, indent=2) + "\n")
print(json.dumps({"status": "L_O_TRAINING_COMPLETE", "runs": runs}, indent=2))
PY
