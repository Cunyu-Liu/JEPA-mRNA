#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
BASE=$W/eval_decision/speed_round36/profile
OUT=$W/eval_decision/speed_round36/optimized
AUDIT=$W/audits/round36_optimization_window
mkdir -p "$OUT" "$AUDIT"
printf 'WAIT_BASELINE time=%s\n' "$(date -Is)" | tee "$AUDIT/window.log"
while ! grep -q 'PROFILE_COMPLETE' "$W/audits/round36_speed_window/window.log" 2>/dev/null; do
  sleep 60
done
for pid in $(pgrep -f '[r]un_nb_secstr_queue_round35.py' || true); do
  args=$(ps -p "$pid" -o args=)
  if [[ "$args" == *"run_nb_secstr_queue_round35.py"* ]]; then
    ps -p "$pid" -o pid,args >> "$AUDIT/stopped_nb_scheduler.txt"
    kill "$pid"
  fi
done
resume_nb() {
  cd /home/rna_jev
  nohup flock -n /tmp/round35_nb_b4.lock "$PY" tools/run_nb_secstr_queue_round35.py --parallel 1 --min-free-mib 8000 --interval 30 >> logs/round35_nb_b4_queue.log 2>&1 &
  printf 'RESUME_NB queue_pid=%s time=%s\n' "$!" | tee -a "$AUDIT/window.log"
}
trap resume_nb EXIT
printf 'WAIT_NB_PROCESSES time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
while ps -eo args= | grep -E '[n]b_secstr_dual_round34.py|[q]a_nb_secstr_round34.py' > /dev/null; do
  sleep 60
done
printf 'WAIT_GPU_EXCLUSIVE time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
  sleep 60
done
nvidia-smi > "$AUDIT/nvidia_smi_exclusive.txt"
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable; refusing CPU fallback'
y = torch.randn((1024, 1024), device='cuda') @ torch.randn((1024, 1024), device='cuda')
torch.cuda.synchronize()
assert y.device.type == 'cuda' and torch.isfinite(y).all()
print('CUDA_ASSERT_PASS')
PY
run_candidate() {
  local model=$1 batch=$2 extra=$3
  printf 'CANDIDATE model=%s batch=%s extra=%s time=%s\n' "$model" "$batch" "$extra" "$(date -Is)" | tee -a "$AUDIT/window.log"
  "$PY" tools/profile_speed_round36.py --model "$model" --batch-size "$batch" --repeats 3 \
    --baseline "$BASE/${model}_b${batch}_profile.json" $extra \
    --out "$OUT/${model}_b${batch}${extra// /_}_profile.json" >> "$AUDIT/candidates.log" 2>&1
}
run_candidate r2d 1 '--bf16-r2d-encoder'
run_candidate r2d 4 '--bf16-r2d-encoder'
run_candidate xens2 1 '--bf16-r2d-encoder'
run_candidate xens2 4 '--bf16-r2d-encoder'
run_candidate plana 16 ''
"$PY" tools/head_only_repro_round36.py --checkpoint "$W/weights/main_ablations/r2d.pt" --batch-size 256 \
  --out "$W/eval_decision/speed_round36/head_only_b256_fp16_repro.json" >> "$AUDIT/candidates.log" 2>&1
printf 'OPTIMIZATION_COMPLETE time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
