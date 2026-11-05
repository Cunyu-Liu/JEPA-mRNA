#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round39_xens2_adaptive_speed.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out=$root/speed_round39/xens2_adaptive_64_32_16_full_ts0.json
log=/home/rna_jev/logs/round39_clean_benchmark/xens2_adaptive_exclusive.log
gpu_count() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
while [[ ! -s "$root/xens2/calibration/independent_verification.json" || ! -s "$root/bpfold/calibration/independent_verification.json" ]]; do
    sleep 120
done
while [[ ! -s "$out" ]]; do
    while [[ "$(gpu_count)" -ne 0 ]]; do
        echo "ADAPTIVE_SPEED_WAIT_EXCLUSIVE $(date -Is)"
        sleep 120
    done
    "$python" tools/speed_xens2_adaptive_round39.py --data /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl \
        --baseline "$root/speed_round39/xens2_b32_full_ts0.json" --plan 256:64,384:32,600:16 \
        --repeats 3 --out "$out" > "$log" 2>&1 &
    pid=$!
    interference=0
    while [[ " $(jobs -pr) " == *" $pid "* ]]; do
        if [[ "$(gpu_count)" -gt 1 ]]; then
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            echo "ADAPTIVE_SPEED_INTERFERENCE pid=$pid $(date -Is)"
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
p=Path(sys.argv[1]);x=json.loads(p.read_text())
assert x["status"]=="PASS_ADAPTIVE_SPEED_CANDIDATE" and len(x["runs"])==3
assert x["protocol"]["device_type"]=="cuda" and x["protocol"]["cached_embeddings"] is False
assert x["protocol"]["n_sequences"]==1288 and x["protocol"]["n_length_filtered"]==0
assert x["pair_identity_fraction_vs_b32"]>=0.99 and x["absolute_f1_delta"]<=0.001
print("ADAPTIVE_SPEED_QA_PASS",x["throughput_median_seq_per_s"],x["speedup_vs_b32"],flush=True)
print("SHA256",hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
PY
