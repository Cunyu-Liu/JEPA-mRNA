#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_pairjepa_gate_smoke.lock
flock -n 9 || { echo "another Pair-JEPA gate smoke is active"; exit 1; }
out="$ROOT/runs/round52_rinalmo_pairjepa_gate_smoke"
if [[ -e "$out/run_meta.json" ]]; then
  echo "refusing to overwrite $out" >&2
  exit 1
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 14000 )); then
  echo "insufficient memory headroom for Pair-JEPA smoke: ${used} MiB" >&2
  exit 42
fi
"$PY" "$REPO/tools/train_plan_a.py" \
  --data "$ROOT/data/validation/vl0_196.jsonl" \
  --teacher-dir "$ROOT/data/teacher_vl0" \
  --out "$out" --steps 1 --batch-size 1 --seed 42 \
  --save-every 1 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round52" \
  --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
  --head-lr 1e-4 --backbone-lr 1e-5 --nll-normalization sum \
  --zchunk 64 --gpu-reserve-gb 2 --scorer pairjepa --pair-latent-dim 490 --ripe-mode r6 \
  --lambda-jepa 0.1 --jepa-target ema --jepa-mask-prob 0.15 \
  --jepa-log-every 25 --jepa-rank-samples 64 \
  > "$ROOT/logs/round52/rinalmo_pairjepa_gate_smoke.log" 2>&1
"$PY" "$REPO/tools/qa_rinalmo_pairjepa_gate_smoke_round52.py" \
  --run "$out" --output "$ROOT/audits/round52/rinalmo_pairjepa_gate_smoke_qa.json"
