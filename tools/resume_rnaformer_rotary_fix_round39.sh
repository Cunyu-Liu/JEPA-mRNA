#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
checksums=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/rnaformer/SHA256SUMS.round39

"$python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/home/rna_jev/tools")
from aggregate_rnaformer_round39 import checksums, validate
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/rnaformer/interfam")
official = checksums(Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/baselines/rnaformer/SHA256SUMS.round39"))
result = validate(root / "raw/ref_pdb_ts1.json", "ref_pdb_ts1", False, official)
assert result["strict_micro_f1"]["values"] == [0.8414634146341463] * 3
print("RNAFORMER_ROTARY_FIX_TS1_GATE_PASS", flush=True)
PY

run_variant() {
    local variant=$1
    bash tools/run_rnaformer_variant_round39.sh "$variant"
    "$python" tools/aggregate_rnaformer_round39.py \
        --root "$root/rnaformer/$variant" --variant "$variant" \
        --official-checksums "$checksums" --out "$root/rnaformer/$variant/raw_clean_summary.json"
}

run_variant interfam
bash tools/run_rnaformer_bprna_round39.sh
"$python" tools/aggregate_rnaformer_round39.py \
    --root "$root/rnaformer" --variant bprna \
    --official-checksums "$checksums" --out "$root/rnaformer/bprna_raw_clean_summary.json"
run_variant intrafam
run_variant biophysical
echo RNAFORMER_ALL_FOUR_VARIANTS_RAW_CLEAN_8_SPLIT_PASS
