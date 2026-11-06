#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
exec 9>/tmp/round39_xens2_stream_speed.lock
flock -n 9 || exit 0
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
out=$root/speed_round39/xens2_two_stream_b32_exclusive.json
baseline=$root/speed_round39/xens2_b32_exclusive.json
while [[ ! -s "$out" ]]; do
    set +e
    "$python" tools/speed_xens2_stream_round39.py \
        --data /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl \
        --baseline "$baseline" --out "$out"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
        break
    fi
    if [[ "$status" -eq 2 ]] && compgen -G "$root/speed_round39/xens2_two_stream_b32_exclusive.interference_*.json" > /dev/null; then
        echo "TWO_STREAM_SPEED_WAIT_EXCLUSIVE $(date -Is)"
        sleep 600
        continue
    fi
    echo "TWO_STREAM_SPEED_REAL_FAILURE status=$status $(date -Is)"
    exit "$status"
done
