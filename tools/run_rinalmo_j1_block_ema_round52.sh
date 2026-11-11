#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round52_rinalmo_j1_block_ema.lock
flock -n 9 || { echo "another J1-block-EMA queue is active"; exit 1; }
until test -f "$ROOT/audits/round52/rinalmo_jmlp_training_qa.json" \
    && test -f "$ROOT/audits/round52/rinalmo_pairjepa_structure_log_smoke_qa.json"; do
  sleep 300
done
base=$("$PY" - "$ROOT/audits/round52/rinalmo_jmlp_training_qa.json" "$ROOT/audits/round52/rinalmo_pairjepa_structure_log_smoke_qa.json" <<'PY'
import json
import sys
qa = json.load(open(sys.argv[1]))
smoke = json.load(open(sys.argv[2]))
assert qa["status"] == "PASS_JMLP_THREE_SEED_TRAINING_QA"
assert qa["allow_j1_block_ema"] and not qa["allow_j1_fixed"] and not qa["allow_j_random"]
assert qa["base"] in ("length", "olr")
assert smoke["status"] == "PASS_PAIRJEPA_STRUCTURE_LOG_ONE_STEP_CUDA_QA"
assert smoke["device"] == "cuda" and not smoke["scientific_conclusion"]
print(qa["base"])
PY
)
if [[ "$base" == length ]]; then normalization=length; backbone_lr=1e-5;
else normalization=sum; backbone_lr=1e-6; fi
for seed in 42 2026 3407; do
  out="$ROOT/runs/round52_rinalmo_j1_block_ema_${base}_s${seed}"
  mkdir -p "$out"
  if "$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" \
      --arm j1_block_ema --seed "$seed" \
      --output "$ROOT/audits/round52/rinalmo_j1_block_ema_${base}_s${seed}_qa.json" \
      > "$ROOT/logs/round52/rinalmo_j1_block_ema_${base}_s${seed}_qa.log" 2>&1; then
    echo "skip validated J1-block-EMA $base $seed"
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
    --lambda-jepa 0.1 --jepa-target ema --jepa-context structured_block \
    --jepa-mask-prob 0.15 --jepa-log-every 25 --jepa-rank-samples 64 \
    >> "$ROOT/logs/round52/rinalmo_j1_block_ema_${base}_s${seed}.log" 2>&1
  "$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" \
    --arm j1_block_ema --seed "$seed" \
    --output "$ROOT/audits/round52/rinalmo_j1_block_ema_${base}_s${seed}_qa.json" \
    > "$ROOT/logs/round52/rinalmo_j1_block_ema_${base}_s${seed}_qa.log" 2>&1
done
"$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" --arm j1_block_ema \
  --output "$ROOT/audits/round52/rinalmo_j1_block_ema_training_qa.json"
"$PY" "$REPO/tools/qa_rinalmo_pairjepa_training_round52.py" --root "$ROOT" --arm j1_block_ema \
  --output "$ROOT/audits/round52/rinalmo_j1_block_ema_training_qa.recheck.json"
cmp "$ROOT/audits/round52/rinalmo_j1_block_ema_training_qa.json" \
    "$ROOT/audits/round52/rinalmo_j1_block_ema_training_qa.recheck.json"
