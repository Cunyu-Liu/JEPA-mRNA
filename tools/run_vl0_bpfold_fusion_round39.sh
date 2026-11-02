#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision
for split in ref_pdb_ts2 ref_pdb_ts1 ref_pdb_ts3 ref_pdb_ts_hard testsetb bprna_ts0 bprna_new archiveii_embok_clean; do
    out=$root/round39_xens2/vl0_fusion/clean/$split.json
    if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
result = json.load(open(sys.argv[1]))
code = hashlib.sha256(Path("tools/evaluate_vl0_bpfold_fusion_round39.py").read_bytes()).hexdigest()
assert result["status"] == "PASS_FUSION_CLEAN_INTERSECTION" and len(result["repeats"]) == 3
assert result["code_sha256"] == code
PY
    then
        continue
    fi
    "$python" tools/evaluate_vl0_bpfold_fusion_round39.py \
        --split "$split" --fit "$root/round39_xens2/vl0_fusion/fit.json" \
        --benchmark "$root/round39_clean_benchmark" \
        --teacher-manifest "$root/round39_xens2/teacher_vl0/manifest.json" \
        --bpfold-checkpoints /teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/round39/BPfold/model_predict \
        --out "$out" > "logs/round39_xens2/fusion_clean_${split}.log" 2>&1
done
echo XENS2_BPFOLD_FUSION_CLEAN_8_SPLIT_PASS
