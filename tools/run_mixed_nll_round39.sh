#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
data=$root/data/round39_mixed_nll/mixed_tr1c_rnastralign_clean.jsonl
audit=$root/data/round39_mixed_nll/audit.json
verification=$root/data/round39_mixed_nll/independent_verification.json
init=$root/eval_decision/round39_xens2/runs/xens2_plana_distill_s0/best.pt
out=$root/eval_decision/round39_mixed_nll/runs/nll_s0
mkdir -p "$out" logs/round39_mixed_nll
"$python" - "$data" "$audit" "$verification" "$init" "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch
data,audit,verification,init,out=map(Path,sys.argv[1:])
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
a=json.loads(audit.read_text()); v=json.loads(verification.read_text())
assert a["status"]=="PASS" and a["out_sha256"]==sha256(data) and a["combined_n"]==44221
assert v["status"]=="PASS_INDEPENDENT_LABEL_AND_LEAKAGE_RECHECK" and v["data_sha256"]==a["out_sha256"]
assert v["additions_n"]==1657 and len(v["per_split"])==9
assert sha256(init)=="2997254f2540e48edc4b42b63d705c470182014519ca9f27c6a9a366a8f0b9f3"
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=18*(1<<30),f"INSUFFICIENT_CUDA_MEMORY_REQUIRED_18_GIB free={free/(1<<30):.3f} total={total/(1<<30):.3f}"
assert not (out/"run_meta.json").exists()
contract={"schema_version":1,"seed":0,"objective":{"nll":1.0,"distill":0.0,"rlcd":0.0,"cal":0.0},
          "data":str(data),"data_sha256":a["out_sha256"],"audit_sha256":sha256(audit),
          "verification_sha256":sha256(verification),"init":str(init),"init_sha256":sha256(init),
          "trainer_sha256":sha256(Path("tools/train_plan_a.py")),
          "launcher_sha256":sha256(Path("tools/run_mixed_nll_round39.sh")),
          "device":torch.cuda.get_device_name(0),"free_gib_at_preflight":free/(1<<30),
          "validation":"bprna_vl0; every=500 patience=5 min_delta=1e-6; steps=1000000 is safety cap only"}
(out/"run_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
print("MIXED_NLL_PREFLIGHT_PASS",contract["free_gib_at_preflight"],flush=True)
PY
"$python" tools/train_plan_a.py \
    --data "$data" --teacher-dir "" \
    --validation-data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl --validation-teacher-dir "" \
    --out "$out" --steps 1000000 --validation-every 500 --validation-patience 5 \
    --validation-min-delta 1e-6 --init-checkpoint "$init" \
    --lambda-nll 1 --lambda-distill 0 --lambda-rlcd 0 --lambda-cal 0 \
    --batch-size 1 --head-lr 1e-4 --backbone-lr 1e-5 --seed 0 \
    --warmup-head-steps 1600 --unfreeze-every 800 --unfreeze-per-step 2 \
    --warmup-steps 5 --save-every 500 --snapshot-every 2000 \
    --snapshot-dir "$root/eval_decision/round39_mixed_nll/snapshots" \
    > logs/round39_mixed_nll/nll_s0.log 2>&1
"$python" - "$out" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); x=json.loads((out/"run_meta.json").read_text())
assert x["status"]=="completed" and x["converged"] and x["early_stopped"]
assert x["environment"]["resolved_device"].startswith("cuda")
assert x["objective"]["lambda_nll"]==1 and x["objective"]["lambda_distill"]==0
assert (out/"best.pt").is_file() and (out/"resume.pt").is_file()
print("MIXED_NLL_SEED0_CONVERGED",x["steps_completed"],x["best_step"],x["best_validation_loss"],flush=True)
PY
