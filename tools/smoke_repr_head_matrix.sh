#!/usr/bin/env bash
# Engineering smoke for the representation/head capacity matrix (NO scientific verdict).
# Four arms x 8 steps (seed0, fresh-head-init) + limit export; polls GPU peak memory.
# Artifacts staged on /home because the /teams quota is currently exceeded (EDQUOT).
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
INIT="$ART/prior_alignment_v1/init/shared_init_seed0.pt"
CONTRACT="$REPO/configs/representation_head_mechanism_4arm_v1.json"
STAGE="${STAGE:-/home/rna_jev_artifacts/representation_head_mechanism_4arm_v1/smoke}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
[[ -s "$INIT" ]] || { echo "missing shared init"; exit 2; }
cd "$REPO"
mkdir -p "$STAGE"
# resource gate: wait until enough GPU memory is free (do not disturb co-tenants)
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [[ "$FREE" -ge "$MIN_FREE_MIB" ]] && break
  echo "[gate] free=${FREE}MiB < ${MIN_FREE_MIB}MiB; waiting 60s"; sleep 60
done
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader > "$STAGE/gpu_before.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
# background peak-memory poller (total GPU used; card is gated free at start)
( peak=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1);
    [[ "$u" -gt "$peak" ]] && peak=$u; echo "$peak" > "$STAGE/peak_mem_mib.txt"; sleep 1; done ) &
POLL=$!
trap 'kill $POLL 2>/dev/null || true' EXIT
declare -A DZ=( [control_matched]=128 [repr_wide]=256 [head_deep]=128 [repr_head]=256 )
declare -A BL=( [control_matched]=4 [repr_wide]=4 [head_deep]=8 [repr_head]=8 )
declare -A HD=( [control_matched]=64 [repr_wide]=64 [head_deep]=128 [repr_head]=128 )
for ARM in control_matched repr_wide head_deep repr_head; do
  echo "=== smoke arm=$ARM d_z=${DZ[$ARM]} blocks=${BL[$ARM]} hidden=${HD[$ARM]} ==="
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --teacher-dir "" \
    --contract "$CONTRACT" --contract-mode smoke --train-turner off \
    --raw-score-arm "$ARM" --shared-init "$INIT" \
    --out "$STAGE/$ARM" --snapshot-dir "$STAGE/$ARM/snapshots" \
    --seed 0 --steps 8 --batch-size 4 --head-lr 1e-4 --backbone-lr 1e-6 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --max-unfrozen-blocks 33 --warmup-steps 5 --grad-clip 1 --nll-normalization length \
    --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
    --lambda-nll 1 --lambda-marginal-bce 0 --marginal-vjp-epsilon 1e-3 \
    --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
    --scorer resnet2d --no-early-stop --save-every 500 --snapshot-every 0 --snapshot-steps "" \
    --d-z "${DZ[$ARM]}" --resnet-blocks "${BL[$ARM]}" --resnet-hidden "${HD[$ARM]}" --head-init fresh \
    --resnet-per-block-checkpoint \
    > "$STAGE/${ARM}.train.log" 2>&1
  test -s "$STAGE/$ARM/snapshots/${ARM}_step8.pt" || { echo "no terminal smoke checkpoint: $ARM"; tail -20 "$STAGE/${ARM}.train.log"; exit 1; }
  "$PY" tools/export_prior_alignment_scores.py \
    --checkpoint "$STAGE/$ARM/snapshots/${ARM}_step8.pt" \
    --train-trajectory train_off --contract-mode smoke \
    --out "$STAGE/$ARM/export_train_off" --limit 4 --resume \
    > "$STAGE/${ARM}.export.log" 2>&1
  "$PY" - "$STAGE/$ARM" "$ARM" <<'PY'
import json, sys, math
from pathlib import Path
run=Path(sys.argv[1]); arm=sys.argv[2]
m=json.loads((run/"run_meta.json").read_text())
logs=[json.loads(l) for l in (run/"train_log.jsonl").read_text().splitlines() if l.strip()]
finite=all(math.isfinite(r["loss"]) and math.isfinite(r["gnorm"]) for r in logs)
print(f"[{arm}] d_z={m.get('d_z')} blocks={m.get('resnet_blocks')} hidden={m.get('resnet_hidden')} "
      f"head_init={m.get('head_init')} status={m.get('status')} steps={m.get('steps_completed')} "
      f"scope={m.get('trainable_parameter_scope')} finite_losses={finite} "
      f"raw_score_arm={m['prior_alignment']['raw_score_arm']}")
assert m.get("d_z")=={"control_matched":128,"repr_wide":256,"head_deep":128,"repr_head":256}[arm]
assert m.get("head_init")=="fresh" and finite
PY
done
kill $POLL 2>/dev/null || true
echo "PEAK_GPU_MEM_MiB=$(cat "$STAGE/peak_mem_mib.txt")"
echo "SMOKE_COMPLETE stage=$STAGE"