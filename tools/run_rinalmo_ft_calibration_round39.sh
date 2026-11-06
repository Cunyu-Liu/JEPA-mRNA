#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
exec 9>/tmp/round39_rinalmo_ft_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
weights=/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt
for variant in bprna intrafam interfam biophysical; do
    while [[ ! -s "$root/rnaformer/$variant/calibration/independent_verification.json" ]]; do
        sleep 120
    done
done
test -s "$weights"
"$python" - <<'PY'
import torch
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=8*(1<<30),f"RINALMO_CUDA_MEMORY_REQUIRED_8_GIB free={free/(1<<30):.3f}"
print("RINALMO_FT_CUDA_PREFLIGHT_PASS",free/(1<<30),flush=True)
PY
out_root=$root/rinalmo_ft/calibration
if [[ ! -s "$out_root/preflight_smoke.json" ]]; then
    "$python" tools/calibrate_rnaformer_round39.py --system rinalmo_ft --variant rinalmo_ft --weights "$weights" \
        --data /root/rna_jepa_data/ss_data/jsonl/ref_pdb_ts1.jsonl --out "$out_root/preflight_smoke.json" \
        --limit 1 --smoke > logs/round39_clean_benchmark/rinalmo_ft_calibration_preflight.log 2>&1
fi
for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
    source_result=$root/rinalmo_ft/clean_tr0/$split.json
    summary=$root/rinalmo_ft/summary_raw_clean.json
    mapfile -t paths < <("$python" - "$source_result" "$summary" "$split" "$weights" <<'PY'
import hashlib,json,sys
from pathlib import Path
source_path,summary_path,split,weights=sys.argv[1:]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=json.loads(Path(source_path).read_text());summary=json.loads(Path(summary_path).read_text())
assert source["status"]=="PASS" and source["protocol"]["device_type"]=="cuda"
assert source["protocol"]["cached_embeddings"] is False and source["protocol"]["repeats"]==3
assert source["protocol"]["n_length_filtered"]==0
assert source["protocol"]["model_report"]["weights_sha256"]==sha(weights)
assert summary["clean_tr0"][split]["result_sha256"]==sha(source_path)
data=source["protocol"]["data"];audit_path=source["protocol"]["clean_audit"]
audit=json.loads(Path(audit_path).read_text())
assert audit["status"]=="PASS" and audit["clean_sha256"]==sha(data)
rows=[line for line in Path(data).read_text().splitlines() if line.strip()]
assert len(rows)==audit["clean_n"]==source["protocol"]["n_sequences"]
print(data);print(audit_path)
PY
    )
    data=${paths[0]}
    audit=${paths[1]}
    out=$out_root/$split.json
    if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert x["status"]=="PASS_NATIVE_PROBABILITY_EVALUATION" and x["system"]=="rinalmo_ft"
assert x["code_sha256"]==sha(Path("tools/calibrate_rnaformer_round39.py"))
assert x["data_sha256"]==sha(Path(x["data"])) and x["pair_arrays_sha256"]==sha(Path(x["pair_arrays"]))
PY
    then
        continue
    fi
    "$python" tools/calibrate_rnaformer_round39.py --system rinalmo_ft --variant rinalmo_ft --weights "$weights" \
        --data "$data" --clean-audit "$audit" --out "$out" \
        > "logs/round39_clean_benchmark/rinalmo_ft_calibration_${split}.log" 2>&1
done
"$python" tools/verify_rnaformer_calibration_round39.py --root "$out_root" --variant rinalmo_ft \
    --summary "$root/rinalmo_ft/summary_raw_clean.json" --out "$out_root/independent_verification.json"
