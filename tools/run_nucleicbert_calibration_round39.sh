#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>/tmp/round39_nucleicbert_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
for system in nucleicbert_ft nucleicbert_frozen; do
    if [[ "$system" == nucleicbert_ft ]]; then
        source_root=$root/nucleicbert_ft/clean_tr1c
    else
        source_root=$root/nucleicbert_frozen/clean
    fi
    out_root=$root/$system/calibration
    for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
        source=$source_root/$split.json
        out=$out_root/$split.json
        if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert x["code_sha256"]==sha(Path("tools/prepare_nucleicbert_calibration_round39.py"))
assert x["source_result_sha256"]==sha(Path(x["source_result"]))
if x["status"]=="EMPTY_AFTER_DELEAK":
    pass
else:
    assert x["status"]=="PASS_NATIVE_PROBABILITY_EVALUATION"
    assert x["pair_arrays_sha256"]==sha(Path(x["pair_arrays"]))
PY
        then
            continue
        fi
        "$python" tools/prepare_nucleicbert_calibration_round39.py --system "$system" --source "$source" --out "$out" \
            > "logs/round39_clean_benchmark/${system}_calibration_${split}.log" 2>&1
    done
    "$python" tools/verify_nucleicbert_calibration_round39.py --root "$out_root" --system "$system" \
        --out "$out_root/independent_verification.json"
    for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
        baseline=$out_root/$split.json
        if [[ "$(basename "$baseline")" == ref_pdb_ts3.json ]] && "$python" - "$baseline" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(x["status"]!="EMPTY_AFTER_DELEAK")
PY
        then
            continue
        fi
        out=$root/calibration_comparisons/$split.xens2_$system.json
        if [[ -s "$out" ]]; then
            continue
        fi
        "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system "$system" \
            --xens2 "$root/xens2/calibration/$split.json" --baseline "$baseline" --out "$out"
    done
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for system in ["nucleicbert_ft","nucleicbert_frozen"]:
 for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
  p=root/"calibration_comparisons"/f"{split}.xens2_{system}.json"
  if not p.exists():
   cells.append({"split":split,"system":system,"status":"EMPTY_AFTER_DELEAK"})
   continue
  x=json.loads(p.read_text())
  cells.append({"split":split,"system":system,"status":x["status"],"matched_n":x["matched_n"],
                "result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
out=root/"calibration_comparisons/nucleicbert_summary.json"
payload={"schema_version":1,"status":"PASS_NUCLEICBERT_COMPARISONS_NOT_ALL_ELIGIBLE","cells":cells,
         "boundary":"original 598-nt protocol retained; auxiliary matched sensitivity only; other baselines pending"}
out.write_text(json.dumps(payload,indent=2)+"\n")
print("NUCLEICBERT_COMPARISONS_QA_PASS",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
