#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
exec 9>/tmp/round39_spotrna_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out_root=$root/spotrna/calibration
for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
    source=$root/spotrna/clean_tr0_tr1/$split.json
    out=$out_root/$split.json
    if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert x["code_sha256"]==sha(Path("tools/prepare_spotrna_calibration_round39.py"))
assert x["source_result_sha256"]==sha(Path(x["source_result"]))
if x["status"]=="EMPTY_AFTER_DELEAK":
    pass
else:
    assert x["status"]=="PASS_NATIVE_PROBABILITY_EVALUATION"
    assert x["pair_arrays_sha256"]==sha(Path(x["pair_arrays"]))
PY
    then
        echo "SPOT_CALIBRATION_SKIP_VALID $split"
        continue
    fi
    "$python" tools/prepare_spotrna_calibration_round39.py --source "$source" --split "$split" --out "$out" \
        > "logs/round39_clean_benchmark/spotrna_calibration_$split.log" 2>&1
done
"$python" - "$out_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
from verify_spotrna_calibration_round39 import validate_result
root=Path(sys.argv[1]);cells=[]
for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
 p=root/(split+".json");x=json.loads(p.read_text())
 if x["status"]=="EMPTY_AFTER_DELEAK":
  cells.append({"split":split,"status":"EMPTY_AFTER_DELEAK","result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
 else:
  cells.append(validate_result(p))
out=root/"independent_verification.json"
payload={"schema_version":1,"status":"PASS_SPOTRNA_EXISTING_PROBABILITY_ASSETS_INDEPENDENT_RECOMPUTE","cells":cells,"comparison_status":"PENDING_ALL_ELIGIBLE_BASELINES_MATCHED_CLEAN_PROBABILITIES"}
out.write_text(json.dumps(payload,indent=2)+"\n")
print("SPOT_CALIBRATION_QA_PASS",hashlib.sha256(out.read_bytes()).hexdigest())
PY
