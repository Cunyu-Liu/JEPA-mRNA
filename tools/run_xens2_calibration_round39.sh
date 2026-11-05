#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round39_xens2_calibration.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out_root=$root/xens2/calibration
speed=$root/speed_round39/xens2_b32_full_ts0.json
mkdir -p "$out_root" logs/round39_xens2
while [[ ! -s "$speed" ]]; do
    sleep 120
done
"$python" - "$root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
board=json.loads((root/"own_clean_leaderboard.json").read_text())
assert board["complete_cell_count"]==board["required_cell_count"]==104
speed=json.loads((root/"speed_round39/xens2_b32_full_ts0.json").read_text())
assert speed["status"]=="PASS" and len(speed["runs"])==3
assert speed["protocol"]["device_type"]=="cuda" and speed["protocol"]["cached_embeddings"] is False
print("CLEAN_BOARD_AND_EXCLUSIVE_SPEED_GATE_PASS",flush=True)
PY
"$python" - <<'PY'
import json
from pathlib import Path
import torch
assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info()
assert free>=16*(1<<30),f"CALIBRATION_CUDA_MEMORY_REQUIRED_16_GIB free={free/(1<<30):.3f}"
root=Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/xens2/calibration")
source=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl")
row=next(line for line in source.read_text().splitlines() if line.strip())
(root/"preflight").mkdir(exist_ok=True)
(root/"preflight/bprna_vl0.jsonl").write_text(row+"\n")
print("CALIBRATION_CUDA_PREFLIGHT_PASS",free/(1<<30),flush=True)
PY
fit=$out_root/bprna_vl0_fit.json
"$python" tools/calibrate_xens2_round39.py --fit-vl0 \
    --data "$out_root/preflight/bprna_vl0.jsonl" --out "$out_root/preflight_smoke.json" \
    > logs/round39_xens2/calibration_preflight.log 2>&1
"$python" tools/calibrate_xens2_round39.py --fit-vl0 \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl --out "$fit" \
    > logs/round39_xens2/calibration_vl0_fit.log 2>&1
for split in ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard testsetb archiveii_embok_clean bprna_ts0 bprna_new; do
    data=$root/clean_splits/tr1c/$split.jsonl
    audit=$root/clean_splits/tr1c/$split.audit.json
    test -s "$data" && test -s "$audit"
    "$python" tools/calibrate_xens2_round39.py --calibration "$fit" --data "$data" \
        --clean-audit "$audit" --out "$out_root/$split.json" \
        > "logs/round39_xens2/calibration_$split.log" 2>&1
done
"$python" tools/verify_xens2_calibration_round39.py --root "$out_root" \
    --out "$out_root/independent_verification.json"
