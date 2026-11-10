#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$ROOT/logs/round52" "$ROOT/audits/round52" "$ROOT/weights/round52"
exec 9>/tmp/round52_rinalmo_r0_smoke.lock
flock -n 9 || { echo "another R0 smoke is active"; exit 1; }
out="$ROOT/runs/round52_rinalmo_r0_smoke_s42"
if [[ -e "$out/run_meta.json" ]]; then
  echo "refusing to overwrite $out" >&2
  exit 1
fi
"$PY" "$REPO/tools/rinalmo_preflight.py" --device cuda \
  > "$ROOT/audits/round52/rinalmo_preflight.json" \
  2> "$ROOT/logs/round52/rinalmo_preflight.log"
"$PY" "$REPO/tools/train_plan_a.py" \
  --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
  --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
  --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
  --validation-teacher-dir "$ROOT/data/teacher_vl0" \
  --out "$out" --steps 2 --batch-size 2 --seed 42 \
  --validation-every 2 --validation-patience 6 --validation-min-delta 1e-6 \
  --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round52" \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
  --zchunk 64 --gpu-reserve-gb 3 --scorer pairmlp --pair-latent-dim 490 --ripe-mode r6 \
  > "$ROOT/logs/round52/rinalmo_r0_smoke_s42.log" 2>&1
"$PY" - "$out" <<'PY'
import json
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
meta = json.loads((root / "run_meta.json").read_text())
assert meta["arm"] == "round52_rinalmo_r0_smoke_s42" and meta["seed"] == 42
assert meta["status"] == "completed" and meta["steps_completed"] == 2
assert meta["stop_reason"] == "step_cap" and not meta["converged"]
assert meta["schedule"]["steps"] == 2 and meta["schedule"]["batch_size"] == 2
assert meta["schedule"]["head_lr"] == 1e-4 and meta["schedule"]["backbone_lr"] == 1e-5
assert meta["validation"]["every"] == 2 and meta["validation"]["patience"] == 6
assert meta["objective"]["nll_normalization"] == "sum"
assert meta["scorer"] == "pairmlp" and meta["pair_latent_dim"] == 490
assert meta["ripe"]["mode"] == "r6" and not meta["ripe"]["encoder"]
assert meta["environment"]["resolved_device"].startswith("cuda")
assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
assert meta["backbone_report"]["d_model"] == 1280 and meta["backbone_report"]["n_layers"] == 33
best = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
assert best["step"] == 2 and best["validation_loss"] == meta["best_validation_loss"]
print(json.dumps({"status": "PASS_R0_TWO_STEP_CUDA_SMOKE", "scientific_conclusion": False,
                  "best_vl0": meta["best_validation_loss"],
                  "backbone": meta["backbone_report"]}, indent=2))
PY
