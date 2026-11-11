#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?j1_fixed or j_random}
if [[ "$ARM" != j1_fixed && "$ARM" != j_random ]]; then
  echo "unsupported target arm: $ARM" >&2
  exit 2
fi
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>"/tmp/round52_rinalmo_${ARM}.lock"
flock -n 9 || { echo "another $ARM queue is active"; exit 1; }
if [[ "$ARM" == j1_fixed ]]; then prerequisite=j1_block_ema; permission=allow_j1_fixed; target=fixed;
else prerequisite=j1_fixed; permission=allow_j_random; target=random; fi
until test -f "$ROOT/audits/round52/rinalmo_${prerequisite}_training_qa.json" \
    && test -f "$ROOT/audits/round52/rinalmo_${prerequisite}_training_qa.recheck.json" \
    && cmp -s "$ROOT/audits/round52/rinalmo_${prerequisite}_training_qa.json" \
       "$ROOT/audits/round52/rinalmo_${prerequisite}_training_qa.recheck.json"; do
  sleep 300
done
base=$("$PY" - "$ROOT/audits/round52/rinalmo_${prerequisite}_training_qa.json" "$permission" <<'PY'
import json
import sys
qa = json.load(open(sys.argv[1]))
assert qa["status"].startswith("PASS_") and qa["status"].endswith("_THREE_SEED_TRAINING_QA")
assert qa[sys.argv[2]] and not qa["scientific_conclusion"]
assert qa["base"] in ("length", "olr")
print(qa["base"])
PY
)
if [[ "$base" == length ]]; then normalization=length; backbone_lr=1e-5;
else normalization=sum; backbone_lr=1e-6; fi
for seed in 42 2026 3407; do
  out="$ROOT/runs/round52_rinalmo_${ARM}_${base}_s${seed}"
  mkdir -p "$out"
  if "$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" \
      --arm "$ARM" --seed "$seed" \
      --output "$ROOT/audits/round52/rinalmo_${ARM}_${base}_s${seed}_qa.json" \
      > "$ROOT/logs/round52/rinalmo_${ARM}_${base}_s${seed}_qa.log" 2>&1; then
    echo "skip validated $ARM $base $seed"
    continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  while (( used > 14000 )); do
    sleep 300
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  done
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
    --zchunk 64 --gpu-reserve-gb 3 --scorer pairjepa --pair-latent-dim 490 --ripe-mode r6 \
    --lambda-jepa 0.1 --jepa-target "$target" --jepa-context structured_block \
    --jepa-mask-prob 0.15 --jepa-log-every 25 --jepa-rank-samples 64 \
    >> "$ROOT/logs/round52/rinalmo_${ARM}_${base}_s${seed}.log" 2>&1
  "$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" \
    --arm "$ARM" --seed "$seed" \
    --output "$ROOT/audits/round52/rinalmo_${ARM}_${base}_s${seed}_qa.json" \
    > "$ROOT/logs/round52/rinalmo_${ARM}_${base}_s${seed}_qa.log" 2>&1
done
"$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" --arm "$ARM" \
  --output "$ROOT/audits/round52/rinalmo_${ARM}_training_qa.json"
"$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" --arm "$ARM" \
  --output "$ROOT/audits/round52/rinalmo_${ARM}_training_qa.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_${ARM}_training_qa.json" \
    "$ROOT/audits/round52/rinalmo_${ARM}_training_qa.recheck.json"
