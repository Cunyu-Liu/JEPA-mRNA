#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
final=$root/rnaformer/biophysical/raw_clean_summary.json
log=/home/rna_jev/logs/round39_remaining_monitor.log
mkdir -p /home/rna_jev/logs
exec >> "$log" 2>&1

echo "MONITOR_WAIT_RNAFORMER_ALL_VARIANTS $(date -Is)"
while [[ ! -s "$final" ]]; do
    sleep 300
done
"$python" - "$final" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x["status"]=="PASS_RAW_CLEAN_8x3_INDEPENDENT_RECOMPUTE"
print("RNAFORMER_ALL_VARIANTS_RAW_CLEAN_CONFIRMED",flush=True)
PY

for attempt in first recompute; do
    out=$root/own_clean_leaderboard.json
    if [[ "$attempt" == recompute ]]; then out=$root/own_clean_leaderboard.recompute.json; fi
    "$python" tools/aggregate_own_clean_leaderboard_round39.py --root "$root" --out "$out"
done
cmp "$root/own_clean_leaderboard.json" "$root/own_clean_leaderboard.recompute.json"
rm "$root/own_clean_leaderboard.recompute.json"
echo "OWN_CLEAN_LEADERBOARD_RECOMPUTE_PASS $(sha256sum "$root/own_clean_leaderboard.json")"

for attempt in first recompute; do
    out=$root/matched_clean_sensitivity.json
    if [[ "$attempt" == recompute ]]; then out=$root/matched_clean_sensitivity.recompute.json; fi
    "$python" tools/aggregate_matched_clean_round39.py --root "$root" --out "$out"
done
cmp "$root/matched_clean_sensitivity.json" "$root/matched_clean_sensitivity.recompute.json"
rm "$root/matched_clean_sensitivity.recompute.json"
echo "MATCHED_CLEAN_SENSITIVITY_RECOMPUTE_PASS $(sha256sum "$root/matched_clean_sensitivity.json")"

gpu_count() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
}
while [[ "$(gpu_count)" -ne 0 ]]; do
    echo "EXCLUSIVE_SPEED_WAIT $(date -Is)"
    sleep 120
done

out=$root/speed_round39/xens2_b32_full_ts0.json
if [[ ! -s "$out" ]]; then
    "$python" tools/speed_end_to_end_4090.py --model xens2 --batch-size 32 --repeats 3 \
        --data /root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl --out "$out" \
        > /home/rna_jev/logs/round39_clean_benchmark/xens2_b32_exclusive.log 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if [[ "$(gpu_count)" -gt 1 ]]; then
            echo "EXCLUSIVE_SPEED_INTERFERENCE pid=$pid $(date -Is)"
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            cp /home/rna_jev/logs/round39_clean_benchmark/xens2_b32_exclusive.log \
               /home/rna_jev/logs/round39_clean_benchmark/xens2_b32_exclusive.interference.log
            exit 1
        fi
        sleep 30
    done
    wait "$pid"
fi
"$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x["status"]=="PASS" and len(x["runs"])==3
assert x["protocol"]["device_type"]=="cuda" and x["protocol"]["cached_embeddings"] is False
assert x["protocol"]["n_length_filtered"]==0 and x["protocol"]["n_sequences"]==1288
assert len({run["predictions_sha256"] for run in x["runs"]})==1
print("XENS2_EXCLUSIVE_SPEED_3_REPEAT_PASS",x["throughput_median_seq_per_s"],x["f1_runs"],flush=True)
PY
sha256sum "$out"

probability_flag=$root/xens2/independent_probabilities_ready.flag
if [[ ! -s "$probability_flag" ]]; then
    echo "CALIBRATION_PENDING_INDEPENDENT_PROBABILITY_ARRAYS $(date -Is)"
    echo "NO_ECE_CLAIM_AND_NO_NEW_TRAINING_CANDIDATE_YET"
    exit 0
fi
echo "CALIBRATION_ASSETS_READY_INDEPENDENT_EVALUATION_REQUIRED"
