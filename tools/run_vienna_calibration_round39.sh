#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export RNA_SITE_PACKAGES=/root/miniconda3/envs/codonflow/lib/python3.10/site-packages
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>/tmp/round39_vienna_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb bprna_ts0 archiveii_embok_clean bprna_new)
for split in "${splits[@]}"; do
    if [[ ! -s "$root/vienna/calibration/$split/vienna_mea.json" ]]; then
        "$python" tools/calibrate_vienna_round39.py \
            --source "$root/vienna/raw_strict/$split.json" --out "$root/vienna/calibration/$split" \
            > "logs/round39_clean_benchmark/vienna_calibration_$split.log" 2>&1
    fi
done
verification=$root/vienna/calibration/independent_verification.json
if [[ ! -s "$verification" ]]; then
    "$python" tools/calibrate_vienna_round39.py \
        --root "$root/vienna/calibration" --out "$verification"
fi
for split in "${splits[@]}"; do
    for system in vienna_mfe vienna_centroid vienna_mea; do
        out=$root/calibration_comparisons/$split.xens2_$system.json
        if [[ ! -s "$out" ]]; then
            "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system "$system" \
                --baseline "$root/vienna/calibration/$split/$system.json" \
                --xens2 "$root/xens2/calibration/$split.json" --out "$out" \
                > "logs/round39_clean_benchmark/vienna_comparison_${split}_$system.log" 2>&1
        fi
    done
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
 for system in ["vienna_mfe","vienna_centroid","vienna_mea"]:
  p=root/"calibration_comparisons"/f"{split}.xens2_{system}.json"
  x=json.loads(p.read_text())
  assert x["status"] in {"PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD","EMPTY_MATCHED_CALIBRATION_INTERSECTION","EMPTY_AFTER_DELEAK"}
  cells.append({"split":split,"system":system,"status":x["status"],"matched_n":x["matched_n"],"result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
payload={"schema_version":1,"status":"PASS_VIENNA_COMPARISONS_NOT_ALL_ELIGIBLE","cells":cells,
         "boundary":"native float64 partition-function BPP; auxiliary matched sensitivity only; other baselines pending"}
out=root/"calibration_comparisons/vienna_summary.json"
out.write_text(json.dumps(payload,indent=2)+"\n")
print("VIENNA_COMPARISONS_QA_PASS",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
