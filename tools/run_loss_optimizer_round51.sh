#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
export ROOT
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=$REPO:$REPO/src:$REPO/tools:$REPO/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>/tmp/round51_loss_optimizer.lock
flock -n 9 || { echo "another L/O queue is active"; exit 1; }
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS_R_REPLAY_THREE_CHECK_QA" and d["allow_l_length_single_factor"] and d["allow_o_lr_single_factor"] and not d["allow_loss_optimizer_interaction"] and not d["allow_capacity_stage"]' "$ROOT/audits/round51/r6_replay_decision.json"
mkdir -p "$ROOT/logs/round51" "$ROOT/audits/round51"

run_one() {
  local arm=$1 seed=$2 normalization backbone_lr out
  if [ "$arm" = length ]; then normalization=length; backbone_lr=1e-5; else normalization=sum; backbone_lr=1e-6; fi
  out=$ROOT/runs/loss_${arm}_s${seed}
  mkdir -p "$out"
  if [ -f "$out/run_meta.json" ] && ARM=$arm SEED=$seed $PY -c 'import json,os; d=json.load(open(os.environ["META"])); assert d["converged"] and d["early_stopped"] and d["status"]=="completed"' META="$out/run_meta.json" 2>/dev/null; then
    echo "skip converged loss_${arm}_s${seed}"
    return 0
  fi
  $PY "$REPO/tools/train_structrfm_round48.py" --pretrained-structrfm "$ROOT/weights/structrfm_round48" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl \
    --teacher-dir /root/rna_jepa_data/ss_data/teacher/bprna_tr1c \
    --validation-data "$ROOT/data/validation/vl0_196.jsonl" \
    --validation-teacher-dir "$ROOT/data/teacher_vl0" \
    --out "$out" --steps 0 --batch-size 2 --seed "$seed" \
    --validation-every 500 --validation-patience 6 --validation-min-delta 1e-6 \
    --save-every 500 --snapshot-every 2000 --snapshot-dir "$ROOT/weights/round51" \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --head-lr 1e-4 --backbone-lr "$backbone_lr" --nll-normalization "$normalization" \
    --zchunk 64 --gpu-reserve-gb 3 --scorer pairmlp --pair-latent-dim 490 --ripe-mode r6 \
    >> "$ROOT/logs/round51/loss_${arm}_s${seed}.log" 2>&1
  CHECKPOINT="$out/best.pt" META="$out/run_meta.json" EXPECTED_ARM="loss_${arm}_s${seed}" SEED=$seed NORMALIZATION=$normalization BACKBONE_LR=$backbone_lr $PY -c 'import hashlib,json,os,torch; m=json.load(open(os.environ["META"])); assert m["converged"] and m["early_stopped"] and m["status"]=="completed"; assert m["arm"]==os.environ["EXPECTED_ARM"] and m["seed"]==int(os.environ["SEED"]); assert m["objective"]["nll_normalization"]==os.environ["NORMALIZATION"] and float(m["schedule"]["backbone_lr"])==float(os.environ["BACKBONE_LR"]); assert m["schedule"]["steps"]==0 and m["validation"]["every"]==500 and m["validation"]["patience"]==6; d=torch.load(os.environ["CHECKPOINT"],map_location="cpu",weights_only=False); assert d["step"]==m["best_step"]; print(json.dumps({"arm":m["arm"],"best_step":d["step"],"best_vl0":m["best_validation_loss"],"checkpoint_sha256":hashlib.sha256(open(os.environ["CHECKPOINT"],"rb").read()).hexdigest()}))'
}

run_group() {
  local pids=() job arm seed fail=0 status
  for job in "$@"; do
    arm=${job%%:*}; seed=${job##*:}
    run_one "$arm" "$seed" & pids+=("$!")
  done
  set +e
  for pid in "${pids[@]}"; do wait "$pid"; status=$?; if [ "$status" -ne 0 ]; then fail=1; fi; done
  set -e
  return "$fail"
}

first=(length:42 length:2026 olr:42)
second=(length:3407 olr:2026 olr:3407)
if ! run_group "${first[@]}"; then
  for job in "${first[@]}"; do run_one "${job%%:*}" "${job##*:}"; done
fi
if ! run_group "${second[@]}"; then
  for job in "${second[@]}"; do run_one "${job%%:*}" "${job##*:}"; done
fi
$PY - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["ROOT"])
runs = {}
for arm in ("length", "olr"):
    expected_norm = "length" if arm == "length" else "sum"
    expected_lr = 1e-5 if arm == "length" else 1e-6
    for seed in (42, 2026, 3407):
        path = root / "runs" / f"loss_{arm}_s{seed}"
        meta = json.loads((path / "run_meta.json").read_text())
        assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
        assert meta["objective"]["nll_normalization"] == expected_norm
        assert float(meta["schedule"]["backbone_lr"]) == expected_lr
        checkpoint = path / "best.pt"
        runs[f"loss_{arm}_s{seed}"] = {
            "best_step": meta["best_step"], "best_vl0": meta["best_validation_loss"],
            "steps_completed": meta["steps_completed"],
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
out = root / "audits" / "round51" / "loss_optimizer_training_complete.json"
out.write_text(json.dumps({"status": "PASS_LOSS_OPTIMIZER_THREE_SEEDS_NATURAL_CONVERGENCE", "runs": runs}, indent=2) + "\n")
print(json.dumps({"status": "complete", "output": str(out), "runs": runs}, indent=2))
PY
