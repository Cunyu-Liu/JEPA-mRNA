#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
"$python" - <<'PY'
import torch
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=12*(1<<30),f"INSUFFICIENT_CUDA_MEMORY_REQUIRED_12_GIB free={free/(1<<30):.3f} total={total/(1<<30):.3f}"
print("RNAFORMER_BPRNA_CUDA_PREFLIGHT_PASS",free/(1<<30),flush=True)
PY
for mode in raw clean; do
    if [[ "$mode" == raw ]]; then
        splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new)
    else
        splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new)
    fi
    for split in "${splits[@]}"; do
        if [[ "$mode" == raw ]]; then
            data=/root/rna_jepa_data/ss_data/jsonl/$split.jsonl
            audit=()
            out=$root/rnaformer/raw/$split.json
        else
            data=$root/clean_splits/rnaformer_bprna_train/splits/$split.jsonl
            audit=(--clean-audit "$root/clean_splits/rnaformer_bprna_train/audits/$split.audit.json")
            out=$root/rnaformer/clean_bprna/$split.json
        fi
        if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert (x["status"]=="PASS" and len(x["runs"])==3 or x["status"]=="EMPTY_AFTER_DELEAK" and not x["runs"])
PY
        then
            continue
        fi
        "$python" tools/speed_end_to_end_4090.py --model rnaformer \
            --batch-size 1 --repeats 3 --allow-long --data "$data" "${audit[@]}" --out "$out" \
            > "logs/round39_clean_benchmark/rnaformer_bprna_${mode}_${split}.log" 2>&1
    done
done
echo RNAFORMER_BPRNA_RAW_CLEAN_8_SPLIT_PASS
