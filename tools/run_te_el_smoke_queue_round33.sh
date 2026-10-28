#!/usr/bin/env bash
set -euo pipefail

ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PYTHON=/root/miniconda3/envs/lucaone/bin/python
DUAL=/home/rna_jev/tools/rinalmo_regression_dual_round33_v4.py
QA=/home/rna_jev/tools/qa_regression_result.py
MIN_FREE_MIB=8000
POLL_SECONDS=120
WAIT_PID="${1:-}"

export CUDA_VISIBLE_DEVICES=0
export REGRESSION_MODULE_PATH="$DUAL"

if [[ -n "$WAIT_PID" ]]; then
    while [[ -d "/proc/$WAIT_PID" ]]; do
        sleep "$POLL_SECONDS"
    done
fi

wait_for_vram() {
    local free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
            echo "Invalid nvidia-smi free-memory reading: $free_mib" >&2
            exit 1
        fi
        if (( free_mib >= MIN_FREE_MIB )); then
            return
        fi
        sleep "$POLL_SECONDS"
    done
}

run_one() {
    local task="$1"
    local cell="$2"
    local name="rinalmo_${task}_${cell}_fold0_round33_smoke_v4"
    local out="$ROOT/eval_decision/$name"
    wait_for_vram
    if [[ ! -f "$out/result.json" ]]; then
        "$PYTHON" "$DUAL" \
            --task "$task" --cell-line "$cell" --fold 0 \
            --device cuda --batch-size 64 --limit 2 --out-dir "$out"
    fi
    "$PYTHON" "$QA" \
        --result "$out/result.json" --task "$task" \
        --cell-line "$cell" --fold 0 --limit 2
    echo "TE_EL_SMOKE_GATE_PASS task=$task cell=$cell"
}

run_one te Muscle
run_one el Muscle
run_one te pc3
run_one el pc3
echo "TE_EL_ALL_REMAINING_SMOKE_GATES_PASS"
