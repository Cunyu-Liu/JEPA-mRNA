#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev/tools${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/home/rna_jev/envs/round39_multimolecule/bin/python
weights=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/round39/multimolecule/spotrna
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out=$root/spotrna
splits=(bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb)
for split in "${splits[@]}"; do
    "$python" tools/benchmark_multimolecule_round39.py \
        --model spotrna --weights "$weights" \
        --data "/root/rna_jepa_data/ss_data/jsonl/$split.jsonl" \
        --out "$out/raw_postprocessed/$split.json" --repeats 3 \
        > "logs/round39_clean_benchmark/spotrna_full_raw_${split}.log" 2>&1
done
for split in "${splits[@]}"; do
    "$python" tools/benchmark_multimolecule_round39.py \
        --model spotrna --weights "$weights" \
        --data "$root/clean_splits/spotrna_tr0_tr1/splits/$split.jsonl" \
        --clean-audit "$root/clean_splits/spotrna_tr0_tr1/audits/$split.audit.json" \
        --out "$out/clean_tr0_tr1/$split.json" --repeats 3 \
        > "logs/round39_clean_benchmark/spotrna_full_clean_${split}.log" 2>&1
done
