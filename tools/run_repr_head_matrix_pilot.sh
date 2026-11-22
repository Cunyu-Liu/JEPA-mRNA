#!/usr/bin/env bash
# GPU-gated pilot for the representation/head capacity mechanism matrix.
# Artifacts staged on /home (the /teams quota is currently exceeded, EDQUOT);
# code/commits remain on /home. Resource gate >=20GB free before each arm so a
# restarting rna-ft-eval co-tenant is never disturbed. No scientific default is
# changed: NLL-only objective, fresh-head-init, MEA temp1/gamma1, no threshold search.
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
ART=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts
SEED="${SEED:-0}"
MODE="${MODE:-pilot}"
STEPS="${STEPS:-9600}"
LIMIT="${LIMIT:-0}"
STAGE="${STAGE:-/home/rna_jev_artifacts/representation_head_mechanism_4arm_v1/seed${SEED}}"
CONTRACT="$REPO/configs/representation_head_mechanism_4arm_v1.json"
INIT="$ART/prior_alignment_v1/init/shared_init_seed${SEED}.pt"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
[[ -s "$INIT" ]] || { echo "missing shared init: $INIT"; exit 2; }
cd "$REPO"
# asset preflight: frozen sha identities
"$PY" - "$CONTRACT" "$INIT" "$SEED" <<'PY' > /tmp/repr_head_preflight.json
import json, sys, hashlib
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
contract=json.loads(Path(sys.argv[1]).read_text()); init=sys.argv[2]; seed=int(sys.argv[3])
tr=contract["training"]
checks={"data_sha256": sha(tr["data"])==tr["data_sha256"],
        "shared_init_sha256": sha(init)==tr["shared_init_sha256_seed0"] if seed==0 else True,
        "decoder_source_sha256": sha(contract["evaluation"]["decoder_source"])==contract["evaluation"]["decoder_source_sha256"],
        "evaluation_data_exists": Path(tr["evaluation_data"]).exists(),
        "contract_frozen": contract["status"]=="FROZEN_BEFORE_PILOT"}
print(json.dumps({"checks":checks,"all_pass":all(checks.values()),
                  "data_sha256":sha(tr["data"]),"shared_init_sha256":sha(init),
                  "decoder_source_sha256":sha(contract["evaluation"]["decoder_source"])},indent=2))
assert all(checks.values()), checks
PY
[[ ! -e "$STAGE" ]] || { echo "refusing to overwrite existing stage: $STAGE"; exit 2; }
mkdir -p "$STAGE"
cp /tmp/repr_head_preflight.json "$STAGE/asset_preflight.json"
cp "$CONTRACT" "$STAGE/contract.executed.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RN_CUDA_INSIDE=1
RUN_GIT_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA256="$(sha256sum tools/train_plan_a_clean_head_p1.py | cut -d' ' -f1)"
INIT_SHA256="$(sha256sum "$INIT" | cut -d' ' -f1)"
CONTRACT_SHA256="$(sha256sum "$CONTRACT" | cut -d' ' -f1)"
gate() { while true; do FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1);
           [[ "$FREE" -ge "$MIN_FREE_MIB" ]] && break; echo "[gate] free=${FREE}MiB < ${MIN_FREE_MIB}MiB; waiting 60s"; sleep 60; done; }
gate
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$STAGE/gpu_before.txt"
declare -A DZ=( [control_matched]=128 [repr_wide]=256 [head_deep]=128 [repr_head]=256 )
declare -A BL=( [control_matched]=4 [repr_wide]=4 [head_deep]=8 [repr_head]=8 )
declare -A HD=( [control_matched]=64 [repr_wide]=64 [head_deep]=128 [repr_head]=128 )
for ARM in control_matched repr_wide head_deep repr_head; do
  gate
  echo "=== arm=$ARM d_z=${DZ[$ARM]} blocks=${BL[$ARM]} hidden=${HD[$ARM]} steps=$STEPS ==="
  "$PY" tools/train_plan_a_clean_head_p1.py \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl --teacher-dir "" \
    --contract "$CONTRACT" --contract-mode "$MODE" --train-turner off \
    --raw-score-arm "$ARM" --shared-init "$INIT" \
    --out "$STAGE/$ARM" --snapshot-dir "$STAGE/$ARM/snapshots" \
    --seed "$SEED" --steps "$STEPS" --batch-size 4 --head-lr 1e-4 --backbone-lr 1e-6 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --max-unfrozen-blocks 33 --warmup-steps 5 --grad-clip 1 --nll-normalization length \
    --abort-loss-threshold 10 --abort-gnorm-threshold 100 \
    --lambda-nll 1 --lambda-marginal-bce 0 --marginal-vjp-epsilon 1e-3 \
    --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 --lambda-jepa 0 \
    --scorer resnet2d --no-early-stop --save-every 500 --snapshot-every 0 --snapshot-steps "" \
    --d-z "${DZ[$ARM]}" --resnet-blocks "${BL[$ARM]}" --resnet-hidden "${HD[$ARM]}" --head-init fresh \
    --resnet-per-block-checkpoint \
    > "$STAGE/${ARM}.train.stdout.log" 2>&1
  test -s "$STAGE/$ARM/snapshots/${ARM}_step${STEPS}.pt" || {
    echo "training stopped before a usable terminal checkpoint: $ARM"; tail -25 "$STAGE/${ARM}.train.stdout.log"; exit 1; }
  "$PY" tools/export_prior_alignment_scores.py \
    --checkpoint "$STAGE/$ARM/snapshots/${ARM}_step${STEPS}.pt" \
    --train-trajectory train_off --contract-mode "$MODE" \
    --out "$STAGE/$ARM/export_train_off" --limit "$LIMIT" --resume \
    > "$STAGE/${ARM}.export.stdout.log" 2>&1
done
"$PY" tools/evaluate_repr_head_matrix.py \
  --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" \
  --stage "$STAGE" --out "$STAGE/evaluation" --limit "$LIMIT" \
  > "$STAGE/evaluation.stdout.log" 2>&1
RUN_GIT_HEAD="$RUN_GIT_HEAD" TRAIN_SHA256="$TRAIN_SHA256" INIT_SHA256="$INIT_SHA256" \
CONTRACT_SHA256="$CONTRACT_SHA256" STAGE="$STAGE" MODE="$MODE" STEPS="$STEPS" SEED="$SEED" \
"$PY" - <<'PY2'
import json, os
from pathlib import Path
stage=Path(os.environ['STAGE'])
verdict=json.loads((stage/'evaluation/pilot_verdict.json').read_text())
prov={k:os.environ[k] for k in ('RUN_GIT_HEAD','TRAIN_SHA256','INIT_SHA256','CONTRACT_SHA256','MODE','STEPS','SEED')}
prov.update({'decision':verdict['decision'],'promoted_arms':verdict['promoted_arms'],
             'external_evaluation_allowed':False,'control_arm':'control_matched',
             'full_control_reused_not_retrained':False,
             'artifact_staging':'/home (teams quota EDQUOT); migratable to /teams'})
(stage/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps(prov,indent=2))
PY2
if [[ "$LIMIT" == "0" ]]; then
  "$PY" tools/verify_repr_head_matrix.py \
    --contract "$CONTRACT" --seed "$SEED" --steps "$STEPS" --stage "$STAGE" \
    --evaluation "$STAGE/evaluation" --out "$STAGE/verification.json" \
    > "$STAGE/verification.stdout.log" 2>&1
  "$PY" tools/independent_repr_head_recheck.py \
    --contract "$CONTRACT" --stage "$STAGE" --evaluation "$STAGE/evaluation" \
    --out "$STAGE/final_independent_check.json" --seed "$SEED" --steps "$STEPS" \
    > "$STAGE/recheck.stdout.log" 2>&1
fi
echo "REPR_HEAD_MATRIX_COMPLETE stage=$STAGE mode=$MODE limit=$LIMIT"