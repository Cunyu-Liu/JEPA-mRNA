#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export NB_CKPT=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
weights=/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/nb_finetuned.pt
for mode in raw clean; do
    for split in ref_pdb_ts2 bprna_ts0 bprna_new archiveii_embok_clean ref_pdb_ts1 ref_pdb_ts3 ref_pdb_ts_hard testsetb; do
        if [[ "$mode" == raw ]]; then
            data=/root/rna_jepa_data/ss_data/jsonl/$split.jsonl
            audit=()
            out=$root/nucleicbert_ft/raw/$split.json
        else
            data=$root/clean_splits/tr1c/$split.jsonl
            audit=(--clean-audit "$root/clean_splits/tr1c/$split.audit.json")
            out=$root/nucleicbert_ft/clean_tr1c/$split.json
        fi
        if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
assert (result["status"] == "PASS" and len(result["repeats"]) == 3 or
        result["status"] == "EMPTY_AFTER_DELEAK" and not result["repeats"])
PY
        then
            continue
        fi
        "$python" tools/benchmark_nucleicbert_round39.py \
            --weights "$weights" --data "$data" "${audit[@]}" --out "$out" --repeats 3 \
            > "logs/round39_clean_benchmark/nucleicbert_ft_${mode}_${split}.log" 2>&1
        if [[ "$mode:$split" == raw:ref_pdb_ts2 ]]; then
            "$python" - "$out" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
f1 = result["repeats"][0]["strict_micro_f1"]
assert abs(f1 - 0.6434540389972144) <= 0.001, f1
PY
        fi
    done
done
echo NUCLEICBERT_FT_RAW_CLEAN_8_SPLIT_PASS
