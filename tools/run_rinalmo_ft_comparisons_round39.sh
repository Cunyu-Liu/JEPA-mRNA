#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>/tmp/round39_rinalmo_ft_comparisons.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
while [[ ! -s "$root/rinalmo_ft/calibration/independent_verification.json" ]]; do
    echo "RINALMO_COMPARISON_WAIT $(date -Is)"
    sleep 120
done
for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
    out=$root/calibration_comparisons/$split.xens2_rinalmo_ft.json
    baseline=$root/rinalmo_ft/calibration/$split.json
    if [[ -s "$out" ]]; then
        continue
    fi
    if "$python" - "$baseline" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(x.get("status")!="EMPTY_AFTER_DELEAK")
PY
    then
        "$python" - "$out" "$baseline" "$root/xens2/calibration/$split.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,baseline,xens2=map(Path,sys.argv[1:])
base=json.loads(baseline.read_text());ours=json.loads(xens2.read_text())
payload={"schema_version":1,"status":"EMPTY_AFTER_DELEAK","matched_n":0,"baseline_system":"rinalmo_ft",
         "baseline_clean_n":0,"xens2_clean_n":ours["n_sequences"],
         "baseline_result_sha256":hashlib.sha256(baseline.read_bytes()).hexdigest(),
         "xens2_result_sha256":hashlib.sha256(xens2.read_bytes()).hexdigest(),
         "calibration":{},"structure_confidence":{}}
out.write_text(json.dumps(payload,indent=2)+"\n")
PY
        continue
    fi
    "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system rinalmo_ft \
        --baseline-summary "$root/rinalmo_ft/summary_raw_clean.json" \
        --baseline "$baseline" --xens2 "$root/xens2/calibration/$split.json" --out "$out" \
        > "logs/round39_clean_benchmark/rinalmo_ft_comparison_$split.log" 2>&1
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
 p=root/"calibration_comparisons"/f"{split}.xens2_rinalmo_ft.json"
 x=json.loads(p.read_text())
 assert x["status"] in {"PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD","EMPTY_MATCHED_CALIBRATION_INTERSECTION","EMPTY_AFTER_DELEAK"}
 cells.append({"split":split,"status":x["status"],"matched_n":x["matched_n"],"result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
payload={"schema_version":1,"status":"PASS_RINALMO_FT_CALIBRATION_COMPARISONS_NOT_ALL_ELIGIBLE","cells":cells,
         "boundary":"auxiliary matched sensitivity; empty intersections never assigned F1=0 or ECE=0; other baselines pending"}
out=root/"calibration_comparisons/rinalmo_ft_summary.json"
out.write_text(json.dumps(payload,indent=2)+"\n")
print("RINALMO_FT_COMPARISONS_QA_PASS",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
