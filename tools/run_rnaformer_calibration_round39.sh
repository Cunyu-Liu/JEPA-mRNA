#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round39_rnaformer_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
weights_root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/rnaformer
while [[ ! -s "$root/speed_round39/xens2_adaptive_64_32_16_full_ts0.json" ]]; do
    sleep 120
done
for variant in bprna intrafam interfam biophysical; do
    case "$variant" in
        bprna) suffix=bprna; corpus=rnaformer_bprna_train ;;
        intrafam) suffix=intra_family_finetuned; corpus=rnaformer_intrafam_train ;;
        interfam) suffix=inter_family_finetuned; corpus=rnaformer_interfam_train ;;
        biophysical) suffix=biophysical; corpus=rnaformer_biophysical_train ;;
    esac
    weights=$weights_root/RNAformer_32M_state_dict_${suffix}.pth
    config=$weights_root/RNAformer_32M_config_${suffix}.yml
    out_root=$root/rnaformer/$variant/calibration
    while ! "$python" - <<'PY'
import torch
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=12*(1<<30)
PY
    do
        sleep 120
    done
    if [[ ! -s "$out_root/preflight_smoke.json" ]]; then
        "$python" tools/calibrate_rnaformer_round39.py --weights "$weights" --config "$config" --variant "$variant" \
            --data /root/rna_jepa_data/ss_data/jsonl/ref_pdb_ts1.jsonl --out "$out_root/preflight_smoke.json" \
            --limit 1 --smoke > "logs/round39_clean_benchmark/rnaformer_${variant}_calibration_preflight.log" 2>&1
    fi
    for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
        data=$root/clean_splits/$corpus/splits/$split.jsonl
        audit=$root/clean_splits/$corpus/audits/$split.audit.json
        out=$out_root/$split.json
        if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if x["status"]=="EMPTY_AFTER_DELEAK":
    pass
else:
    assert x["status"]=="PASS_NATIVE_PROBABILITY_EVALUATION" and x["code_sha256"]==sha(Path("tools/calibrate_rnaformer_round39.py"))
    assert x["data_sha256"]==sha(Path(x["data"])) and x["pair_arrays_sha256"]==sha(Path(x["pair_arrays"]))
PY
        then
            continue
        fi
        "$python" tools/calibrate_rnaformer_round39.py --weights "$weights" --config "$config" --variant "$variant" \
            --data "$data" --clean-audit "$audit" --out "$out" \
            > "logs/round39_clean_benchmark/rnaformer_${variant}_calibration_${split}.log" 2>&1
    done
    "$python" tools/verify_rnaformer_calibration_round39.py --root "$out_root" --variant "$variant" \
        --summary "$root/rnaformer/$variant/raw_clean_summary.json" --out "$out_root/independent_verification.json"
done
