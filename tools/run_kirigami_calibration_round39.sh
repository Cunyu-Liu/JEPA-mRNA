#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec 9>/tmp/round39_kirigami_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb bprna_ts0 archiveii_embok_clean bprna_new)
for split in "${splits[@]}"; do
    out=$root/kirigami_fast/calibration/$split.json
    if [[ ! -s "$out" ]]; then
        "$python" tools/calibrate_kirigami_round39.py \
            --source "$root/kirigami_fast/clean_tr0/$split.json" --out "$out" \
            > "logs/round39_clean_benchmark/kirigami_calibration_$split.log" 2>&1
    fi
done
verification=$root/kirigami_fast/calibration/independent_verification.json
if [[ ! -s "$verification" ]]; then
    "$python" tools/calibrate_kirigami_round39.py \
        --root "$root/kirigami_fast/calibration" --out "$verification"
fi
for split in "${splits[@]}"; do
    out=$root/calibration_comparisons/$split.xens2_kirigami.json
    if [[ ! -s "$out" ]]; then
        "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system kirigami \
            --baseline "$root/kirigami_fast/calibration/$split.json" \
            --xens2 "$root/xens2/calibration/$split.json" --out "$out" \
            > "logs/round39_clean_benchmark/kirigami_comparison_$split.log" 2>&1
    fi
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
 p=root/"calibration_comparisons"/f"{split}.xens2_kirigami.json"
 x=json.loads(p.read_text())
 assert x["status"] in {"PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD","EMPTY_MATCHED_CALIBRATION_INTERSECTION","EMPTY_AFTER_DELEAK"}
 cells.append({"split":split,"status":x["status"],"matched_n":x["matched_n"],"result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
payload={"schema_version":1,"status":"PASS_KIRIGAMI_COMPARISONS_NOT_ALL_ELIGIBLE","cells":cells,
         "boundary":"native pseudoknot decode retained; auxiliary matched sensitivity only; other baselines pending"}
out=root/"calibration_comparisons/kirigami_summary.json"
out.write_text(json.dumps(payload,indent=2)+"\n")
print("KIRIGAMI_COMPARISONS_QA_PASS",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
