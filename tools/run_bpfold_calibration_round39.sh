#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
exec 9>/tmp/round39_bpfold_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out_root=$root/bpfold/calibration
weights=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/round39/BPfold/model_predict
mkdir -p logs/round39_clean_benchmark
"$python" - <<'PY'
import torch
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=6*(1<<30),f"BPFOLD_CUDA_MEMORY_REQUIRED_6_GIB free={free/(1<<30):.3f}"
print("BPFOLD_CALIBRATION_CUDA_PREFLIGHT_PASS",free/(1<<30),flush=True)
PY
for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
    data=$root/clean_splits/bpfold_train_union/splits/$split.jsonl
    audit=$root/clean_splits/bpfold_train_union/audits/$split.audit.json
    out=$out_root/$split.json
    test -s "$data" && test -s "$audit"
    if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
a=Path(x["pair_arrays"])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert x["status"]=="PASS_NATIVE_PROBABILITY_EVALUATION" and x["code_sha256"]==sha(Path("tools/calibrate_bpfold_round39.py"))
assert x["data_sha256"]==sha(Path(x["data"])) and x["pair_arrays_sha256"]==sha(a)
assert len(x["f1_runs"])==3 and Path(x["official_contact_map_root"]).is_dir()
PY
    then
        echo "BPFOLD_CALIBRATION_SKIP_VALID $split"
        continue
    fi
    "$python" tools/calibrate_bpfold_round39.py --checkpoints "$weights" --data "$data" \
        --clean-audit "$audit" --out "$out" --artifact-root "$out_root/$split" \
        > "logs/round39_clean_benchmark/bpfold_calibration_$split.log" 2>&1
done
"$python" tools/verify_bpfold_calibration_round39.py --root "$out_root" \
    --out "$out_root/independent_verification.json"
