#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
weights=$root/weights/round39/NucleicBERT/secstr.ckpt
out_root=$root/eval_decision/round39_clean_benchmark/nucleicbert_frozen
"$python" - "$weights" <<'PY'
import hashlib,sys,torch
assert torch.cuda.is_available()
h=hashlib.sha256()
with open(sys.argv[1],"rb") as handle:
    for block in iter(lambda:handle.read(8<<20),b""):
        h.update(block)
assert h.hexdigest()=="90b9af474693db3879830a455f4c25662d672dc0f4e9cdc61b90055e9b6d5e07"
print("NUCLEICBERT_FROZEN_CUDA_AND_WEIGHTS_PREFLIGHT_PASS",flush=True)
PY
for split in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
    out=$out_root/raw/$split.json
    if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x["status"]=="PASS" and len(x["repeats"])==3
assert x["system"]=="nucleicbert_frozen_official" and x["batch_size"]==2
assert x["weights_sha256"]=="90b9af474693db3879830a455f4c25662d672dc0f4e9cdc61b90055e9b6d5e07"
PY
    then
        continue
    fi
    "$python" tools/benchmark_nucleicbert_frozen_round39.py \
        --weights "$weights" --data /root/rna_jepa_data/ss_data/jsonl/$split.jsonl \
        --repeats 3 --out "$out" > logs/round39_clean_benchmark/nucleicbert_frozen_raw_${split}.log 2>&1
done
"$python" tools/aggregate_nucleicbert_frozen_round39.py \
    --root "$out_root" --out "$out_root/raw_summary.json"
echo NUCLEICBERT_FROZEN_RAW_8_SPLIT_PASS
