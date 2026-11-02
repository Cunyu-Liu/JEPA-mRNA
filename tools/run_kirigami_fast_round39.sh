#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
weights=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/kirigami/main.ckpt
for split in archiveii_embok_clean ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb; do
    "$python" tools/benchmark_kirigami_round39.py --weights "$weights" \
        --data "/root/rna_jepa_data/ss_data/jsonl/$split.jsonl" \
        --out "$root/kirigami_fast/raw/$split.json" --repeats 3 \
        > "logs/round39_clean_benchmark/kirigami_fast_raw_${split}.log" 2>&1
done
for split in bprna_ts0 bprna_new archiveii_embok_clean ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb; do
    "$python" tools/benchmark_kirigami_round39.py --weights "$weights" \
        --data "$root/clean_splits/tr0/$split.jsonl" \
        --clean-audit "$root/clean_splits/tr0/$split.audit.json" \
        --out "$root/kirigami_fast/clean_tr0/$split.json" --repeats 3 \
        > "logs/round39_clean_benchmark/kirigami_fast_clean_${split}.log" 2>&1
done
echo KIRIGAMI_FAST_REMAINING_RAW_CLEAN_PASS
