#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
OUT=$W/eval_decision/speed_round36/profile
AUDIT=$W/audits/round36_speed_window
mkdir -p "$OUT" "$AUDIT"
TRAINER_PID=${1:?usage: run_round36_speed_window.sh OFFICIAL_S1_TRAINER_PID}
printf 'WAIT trainer_pid=%s time=%s\n' "$TRAINER_PID" "$(date -Is)" | tee "$AUDIT/window.log"
while kill -0 "$TRAINER_PID" 2>/dev/null; do
  sleep 60
done
RUN_JSON=$W/eval_decision/nb_secstr_round35_b4/official_s1/run.json
resume_nb() {
  cd /home/rna_jev
  nohup flock -n /tmp/round35_nb_b4.lock "$PY" tools/run_nb_secstr_queue_round35.py --parallel 1 --min-free-mib 8000 --interval 30 >> logs/round35_nb_b4_queue.log 2>&1 &
  printf 'RESUME_NB queue_pid=%s time=%s\n' "$!" | tee -a "$AUDIT/window.log"
}
trap resume_nb EXIT
if [[ ! -s "$RUN_JSON" ]]; then
  printf 'STOP missing_official_s1_run_json time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
  exit 2
fi
nvidia-smi > "$AUDIT/nvidia_smi_exclusive.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'STOP gpu_not_exclusive time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
  exit 3
fi
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable; refusing CPU fallback'
x = torch.randn((1024, 1024), device='cuda')
y = x @ x
torch.cuda.synchronize()
assert y.device.type == 'cuda' and torch.isfinite(y).all()
print('CUDA_ASSERT_PASS', torch.cuda.get_device_name(0))
PY
for model in r2d plana xens2; do
  for batch in 1 4; do
    printf 'START model=%s batch=%s time=%s\n' "$model" "$batch" "$(date -Is)" | tee -a "$AUDIT/window.log"
    "$PY" tools/profile_speed_round36.py --model "$model" --batch-size "$batch" --repeats 3 \
      --out "$OUT/${model}_b${batch}_profile.json" >> "$AUDIT/profile.log" 2>&1
  done
done
printf 'PROFILE_COMPLETE time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
