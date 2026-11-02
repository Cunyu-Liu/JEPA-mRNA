#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
weights_root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/rnaformer
variant=${1:?usage: run_rnaformer_variant_round39.sh interfam|intrafam|biophysical}
case "$variant" in
    interfam)
        suffix=inter_family_finetuned
        clean_corpus=rnaformer_interfam_train
        ;;
    intrafam)
        suffix=intra_family_finetuned
        clean_corpus=rnaformer_intrafam_train
        ;;
    biophysical)
        suffix=biophysical
        clean_corpus=rnaformer_biophysical_train
        ;;
    *) echo "UNKNOWN_VARIANT $variant" >&2; exit 1 ;;
esac
weights=$weights_root/RNAformer_32M_state_dict_${suffix}.pth
config=$weights_root/RNAformer_32M_config_${suffix}.yml
test -s "$weights" && test -s "$config"
"$python" - <<'PY'
import torch
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=21*(1<<30),f"INSUFFICIENT_CUDA_MEMORY_REQUIRED_21_GIB free={free/(1<<30):.3f} total={total/(1<<30):.3f}"
print("RNAFORMER_VARIANT_CUDA_PREFLIGHT_PASS",free/(1<<30),flush=True)
PY
for mode in raw clean; do
    for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
        if [[ "$mode" == raw ]]; then
            data=/root/rna_jepa_data/ss_data/jsonl/$split.jsonl
            audit=()
            out=$root/rnaformer/$variant/raw/$split.json
        else
            data=$root/clean_splits/$clean_corpus/splits/$split.jsonl
            audit=(--clean-audit "$root/clean_splits/$clean_corpus/audits/$split.audit.json")
            out=$root/rnaformer/$variant/clean/$split.json
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
            --rnaformer-weights "$weights" --rnaformer-config "$config" \
            --batch-size 1 --repeats 3 --allow-long --data "$data" "${audit[@]}" --out "$out" \
            > "logs/round39_clean_benchmark/rnaformer_${variant}_${mode}_${split}.log" 2>&1
    done
done
echo "RNAFORMER_${variant}_RAW_CLEAN_8_SPLIT_PASS"
