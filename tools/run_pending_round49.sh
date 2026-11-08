#!/usr/bin/env bash
set -uo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
cd "$REPO"
mkdir -p "$ROOT/logs/round48"
PIDS=()
NAMES=()
launch() {
  local name=$1 lock=$2 script=$3 log=$4
  flock "/tmp/${lock}.lock" bash "tools/${script}" >> "$ROOT/logs/round48/${log}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  printf '%s pid=%s\n' "$name" "$!"
}
launch ripe_decision round48_ripe_decision run_ripe_decision_round48.sh ripe_decision_queue.log
launch pairjepa_queue round48_pairjepa_queue run_pairjepa_round48.sh pairjepa_queue.log
launch pairjepa_decision round48_pairjepa_decision run_pairjepa_decision_round48.sh pairjepa_decision_queue.log
launch complex_d1 rna_complex_d1 run_complex_d1_round49.sh complex_d1_queue.log
launch complex_decision rna_complex_decision run_complex_decision_round49.sh complex_decision_queue.log
launch complex_teacher rna_complex_teacher run_complex_teacher_round49.sh complex_teacher_queue.log
launch lineardesign round48_lineardesign run_lineardesign_round49.sh lineardesign_queue.log
launch speed round48_speed run_speed_round49.sh speed_queue.log
launch external_speed round48_external_speed run_external_speed_round49.sh external_speed_queue.log
launch remaining_speed round48_remaining_speed run_remaining_speed_round49.sh remaining_speed_queue.log
launch speed_summary round48_speed_summary run_speed_summary_round49.sh speed_summary_queue.log
launch confidence_independent round48_confidence_independent run_confidence_independent_round49.sh confidence_independent_queue.log
failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    status=0
  else
    status=$?
    failed=1
  fi
  printf '%s exit=%s\n' "${NAMES[$index]}" "$status"
done
exit "$failed"
