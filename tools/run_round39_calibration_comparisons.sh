#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>/tmp/round39_calibration_comparisons.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
for split in archiveii_embok_clean bprna_ts0 bprna_new; do
    while [[ ! -s "$root/xens2/calibration/$split.json" || ! -s "$root/bpfold/calibration/$split.json" ]]; do
        echo "CALIBRATION_COMPARISON_WAIT $split $(date -Is)"
        sleep 120
    done
    for system in bpfold spotrna; do
        baseline=$root/$system/calibration/$split.json
        out=$root/calibration_comparisons/$split.xens2_$system.json
        if [[ "$system" == spotrna ]] && [[ "$(basename "$split")" == ref_pdb_ts3 ]]; then
            continue
        fi
        if [[ -s "$out" ]]; then
            continue
        fi
        if [[ "$system" == spotrna ]]; then
            "$python" tools/compare_bpfold_xens2_calibration_round39.py --baseline-system spotrna \
                --xens2 "$root/xens2/calibration/$split.json" --baseline "$baseline" --out "$out"
        else
            "$python" tools/compare_bpfold_xens2_calibration_round39.py \
                --xens2 "$root/xens2/calibration/$split.json" --baseline "$baseline" --out "$out"
        fi
    done
done
"$python" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);cells=[]
for split in ["bprna_ts0","bprna_new","ref_pdb_ts1","ref_pdb_ts2","ref_pdb_ts3","ref_pdb_ts_hard","archiveii_embok_clean","testsetb"]:
 for system in ["bpfold","spotrna"]:
  if split=="ref_pdb_ts3" and system=="spotrna":
   cells.append({"split":split,"baseline":system,"status":"EMPTY_AFTER_DELEAK"})
   continue
  p=root/"calibration_comparisons"/f"{split}.xens2_{system}.json"
  x=json.loads(p.read_text())
  assert x["status"]=="PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD"
  cells.append({"split":split,"baseline":system,"status":x["status"],"matched_n":x["matched_n"],
                "xens2_clean_n":x["xens2_clean_n"],"baseline_clean_n":x.get("baseline_clean_n",x["bpfold_clean_n"]),
                "result_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
out=root/"calibration_comparisons/summary.json"
payload={"schema_version":1,"status":"PASS_BPFOLD_SPOTRNA_CALIBRATION_COMPARISONS_NOT_ALL_ELIGIBLE",
         "cells":cells,"boundary":"auxiliary matched sensitivity only; native own-clean leaderboard primary; other probability baselines pending"}
out.write_text(json.dumps(payload,indent=2)+"\n")
print("CALIBRATION_COMPARISONS_QA_PASS",len(cells),hashlib.sha256(out.read_bytes()).hexdigest())
PY
