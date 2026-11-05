#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round39_xens2_exclusive_speed.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out=$root/speed_round39/xens2_b32_full_ts0.json
log=/home/rna_jev/logs/round39_clean_benchmark/xens2_b32_exclusive.log
gpu_count() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
}
while [[ ! -s "$out" ]]; do
    while [[ "$(gpu_count)" -ne 0 ]]; do
        echo "EXCLUSIVE_SPEED_RETRY_WAIT $(date -Is)"
        sleep 120
    done
    "$python" tools/speed_end_to_end_4090.py --model xens2 --batch-size 32 --repeats 3 \
        --data /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl --out "$out" \
        > "$log" 2>&1 &
    pid=$!
    interference=0
    while [[ " $(jobs -pr) " == *" $pid "* ]]; do
        if [[ "$(gpu_count)" -gt 1 ]]; then
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            echo "EXCLUSIVE_SPEED_RETRY_INTERFERENCE pid=$pid $(date -Is)"
            interference=1
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            cp "$log" "${log%.log}.interference.$stamp.log"
            break
        fi
        sleep 30
    done
    if [[ "$interference" == 1 ]]; then
        continue
    fi
    if [[ -s "$out" ]]; then
        wait "$pid"
        break
    fi
    if ! wait "$pid"; then
        cp "$log" "${log%.log}.failed.$(date -u +%Y%m%dT%H%M%SZ).log"
        exit 1
    fi
done
"$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1])
x=json.loads(path.read_text())
assert x["status"]=="PASS" and len(x["runs"])==3
assert x["protocol"]["device_type"]=="cuda" and x["protocol"]["cached_embeddings"] is False
assert x["protocol"]["n_length_filtered"]==0 and x["protocol"]["n_sequences"]==1288
assert len({run["predictions_sha256"] for run in x["runs"]})==1
print("XENS2_EXCLUSIVE_SPEED_3_REPEAT_PASS",x["throughput_median_seq_per_s"],x["f1_runs"],flush=True)
print("SHA256",hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
PY
