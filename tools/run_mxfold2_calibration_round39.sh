#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec 9>/tmp/round39_mxfold2_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
splits=(ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb bprna_ts0 archiveii_embok_clean bprna_new)
for split in "${splits[@]}"; do
    out=$root/mxfold2/calibration/$split.json
    if [[ ! -s "$out" ]]; then
        "$python" tools/calibrate_mxfold2_round39.py \
            --source "$root/mxfold2/clean_trainsetab/$split.json" --out "$out" \
            > "logs/round39_clean_benchmark/mxfold2_calibration_$split.log" 2>&1
    fi
done
verification=$root/mxfold2/calibration/independent_verification.json
if [[ ! -s "$verification" ]]; then
    "$python" tools/calibrate_mxfold2_round39.py \
        --root "$root/mxfold2/calibration" --out "$verification"
fi
for split in "${splits[@]}"; do
    out=$root/calibration_comparisons/$split.xens2_mxfold2.json
    baseline=$root/mxfold2/calibration/$split.json
    if [[ ! -s "$out" ]]; then
        if "$python" - "$baseline" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(x.get("status")!="FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE")
PY
        then
            "$python" - "$out" "$baseline" "$root/xens2/calibration/$split.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,baseline,xens2=map(Path,sys.argv[1:])
base=json.loads(baseline.read_text());ours=json.loads(xens2.read_text())
payload={"schema_version":1,"status":"NATIVE_BPP_INVALID_NO_CALIBRATION","split":Path(base["data"]).stem,
         "matched_n":None,"baseline_system":"mxfold2","baseline_clean_n":base["n_sequences"],
         "xens2_clean_n":ours["n_sequences"],"baseline_result_sha256":hashlib.sha256(baseline.read_bytes()).hexdigest(),
         "xens2_result_sha256":hashlib.sha256(xens2.read_bytes()).hexdigest(),
         "source_failure":base["failure"],"calibration":{},"structure_confidence":{},
         "note":"native structure F1 retained in own-clean leaderboard; native BPP invalid, no ECE or matched-n substitution"}
out.write_text(json.dumps(payload,indent=2)+"\n")
print("NATIVE_BPP_INVALID_NO_CALIBRATION",out)
PY
        else
            "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system mxfold2 \
                --baseline "$baseline" --xens2 "$root/xens2/calibration/$split.json" --out "$out" \
                > "logs/round39_clean_benchmark/mxfold2_comparison_$split.log" 2>&1
        fi
    fi
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
 p=root/"calibration_comparisons"/f"{split}.xens2_mxfold2.json"
 x=json.loads(p.read_text())
 assert x["status"] in {"PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD","EMPTY_MATCHED_CALIBRATION_INTERSECTION","EMPTY_AFTER_DELEAK","NATIVE_BPP_INVALID_NO_CALIBRATION"}
 cells.append({"split":split,"status":x["status"],"matched_n":x["matched_n"],"result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
payload={"schema_version":1,"status":"COMPLETE_MXFOLD2_COMPARISONS_7_VALID_1_NATIVE_BPP_INVALID","cells":cells,
         "boundary":"seven valid native probability splits; ArchiveII native BPP above one in scored mask, no ECE; auxiliary matched sensitivity only"}
out=root/"calibration_comparisons/mxfold2_summary.json"
out.write_text(json.dumps(payload,indent=2)+"\n")
print("MXFOLD2_COMPARISONS_QA_COMPLETE",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
