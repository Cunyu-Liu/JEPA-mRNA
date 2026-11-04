#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round39_rnaformer_biophysical.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
sums=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/rnaformer/SHA256SUMS.round39
summary=$root/rnaformer/biophysical/raw_clean_summary.json
if [[ ! -s "$summary" ]]; then
    bash tools/run_rnaformer_variant_round39.sh biophysical
    "$python" tools/aggregate_rnaformer_round39.py --root "$root/rnaformer/biophysical" \
        --variant biophysical --official-checksums "$sums" --out "$summary"
fi
"$python" - "$summary" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "PASS_RAW_CLEAN_8x3_INDEPENDENT_RECOMPUTE"
assert len(result["raw"]) == len(result["clean"]) == 8
print("RNAFORMER_BIOPHYSICAL_RAW_CLEAN_CONFIRMED", flush=True)
PY
