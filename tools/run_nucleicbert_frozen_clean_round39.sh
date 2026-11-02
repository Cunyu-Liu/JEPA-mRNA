#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
train=$root/data/nucleicbert_secondary_structure_round34/jsonl/train_rnalign_tr0.jsonl
clean_root=$root/eval_decision/round39_clean_benchmark/clean_splits/nucleicbert_rnastralign_tr0
weights=$root/weights/round39/NucleicBERT/secstr.ckpt
out_root=$root/eval_decision/round39_clean_benchmark/nucleicbert_frozen
"$python" - "$train" <<'PY'
import hashlib,json,sys
h=hashlib.sha256()
with open(sys.argv[1],"rb") as handle:
    for block in iter(lambda:handle.read(1<<20),b""):
        h.update(block)
assert h.hexdigest()=="948729bd9a2af887ee81ff9553dffa8181f8747e1e95dd853c4156621e4ea0d8"
assert sum(1 for line in open(sys.argv[1]) if line.strip())==47963
print("NUCLEICBERT_OFFICIAL_TRAIN_UNION_IDENTITY_PASS",flush=True)
PY
for split in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
    data=/root/rna_jepa_data/ss_data/jsonl/$split.jsonl
    out=$clean_root/splits/$split.jsonl
    audit=$clean_root/audits/$split.audit.json
    work=/home/rna_jev/scratch/round39_nucleicbert_frozen_clean/$split
    if [[ ! -s "$out" || ! -s "$audit" ]]; then
        "$python" tools/build_round39_clean_split.py --train "$train" --split "$data" \
            --out "$out" --audit "$audit" --work-dir "$work" \
            > logs/round39_clean_benchmark/nucleicbert_frozen_clean_build_${split}.log 2>&1
    fi
    "$python" - "$data" "$out" "$audit" <<'PY'
import hashlib,json,sys
raw,clean,audit_path=sys.argv[1:]
a=json.load(open(audit_path))
h=lambda p:hashlib.sha256(open(p,"rb").read()).hexdigest()
n=lambda p:sum(1 for line in open(p) if line.strip())
assert a["status"]=="PASS" and a["split_sha256"]==h(raw) and a["clean_sha256"]==h(clean)
assert a["raw_n"]==n(raw) and a["clean_n"]==n(clean)
assert a["raw_n"]==a["clean_n"]+a["removed_exact"]+a["removed_homology_only"]
assert a["train_sha256"]=="948729bd9a2af887ee81ff9553dffa8181f8747e1e95dd853c4156621e4ea0d8"
print("CLEAN_SPLIT_PASS",raw,a["clean_n"],flush=True)
PY
done
echo NUCLEICBERT_FROZEN_CLEAN_SPLITS_8_PASS
exec flock /tmp/rna_round39_fusion_clean.lock bash -c '
    set -euo pipefail
    cd /home/rna_jev
    export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python=/root/miniconda3/envs/lucaone/bin/python
    root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
    clean_root=$root/eval_decision/round39_clean_benchmark/clean_splits/nucleicbert_rnastralign_tr0
    weights=$root/weights/round39/NucleicBERT/secstr.ckpt
    out_root=$root/eval_decision/round39_clean_benchmark/nucleicbert_frozen
    for split in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
        out=$out_root/clean/$split.json
        if [[ -s "$out" ]] && "$python" - "$out" <<'"'"'PY'"'"'
import json,sys
x=json.load(open(sys.argv[1]))
assert (x["status"]=="PASS" and len(x["repeats"])==3 or x["status"]=="EMPTY_AFTER_DELEAK")
assert x["system"]=="nucleicbert_frozen_official" and x.get("input_max_length_with_special_tokens",600)==600
PY
        then
            continue
        fi
        "$python" tools/benchmark_nucleicbert_frozen_round39.py --weights "$weights" \
            --data "$clean_root/splits/$split.jsonl" --clean-audit "$clean_root/audits/$split.audit.json" \
            --repeats 3 --out "$out" > logs/round39_clean_benchmark/nucleicbert_frozen_clean_${split}.log 2>&1
    done
'
"$python" tools/aggregate_nucleicbert_frozen_round39.py \
    --root "$out_root" --out "$out_root/raw_clean_summary.json"
echo NUCLEICBERT_FROZEN_CLEAN_8_SPLIT_PASS
