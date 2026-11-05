#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>/tmp/round39_rnaformer_comparisons.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
for variant in bprna intrafam interfam biophysical; do
    if [[ "$variant" == bprna ]]; then
        summary=$root/rnaformer/bprna_raw_clean_summary.json
    else
        summary=$root/rnaformer/$variant/raw_clean_summary.json
    fi
    while [[ ! -s "$root/rnaformer/$variant/calibration/independent_verification.json" ]]; do
        echo "RNAFORMER_COMPARISON_WAIT $variant $(date -Is)"
        sleep 120
    done
    for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
        out=$root/calibration_comparisons/$split.xens2_${variant}.json
        if [[ -s "$out" ]]; then
            continue
        fi
        "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system rnaformer \
            --baseline-summary "$summary" --xens2 "$root/xens2/calibration/$split.json" \
            --baseline "$root/rnaformer/$variant/calibration/$split.json" --out "$out"
    done
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for variant in ["bprna","intrafam","interfam","biophysical"]:
 for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
  p=root/"calibration_comparisons"/f"{split}.xens2_{variant}.json"
  x=json.loads(p.read_text())
  assert x["status"] in {"PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD","EMPTY_MATCHED_CALIBRATION_INTERSECTION"}
  cells.append({"variant":variant,"split":split,"status":x["status"],"matched_n":x["matched_n"],
                "result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
out=root/"calibration_comparisons/rnaformer_all_variants_summary.json"
payload={"schema_version":1,"status":"PASS_RNAFORMER_CALIBRATION_COMPARISONS_NOT_ALL_ELIGIBLE","cells":cells,
         "boundary":"auxiliary matched sensitivity; empty intersections never assigned F1=0 or ECE=0; other baselines pending"}
out.write_text(json.dumps(payload,indent=2)+"\n")
print("RNAFORMER_COMPARISONS_QA_PASS",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
