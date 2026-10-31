#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
OUT=$W/eval_decision/speed_round36/baselines
AUDIT=$W/audits/round36_baseline_window
mkdir -p "$OUT" "$AUDIT"
printf 'WAIT_OPTIMIZATION time=%s\n' "$(date -Is)" | tee "$AUDIT/window.log"
while ! grep -q 'OPTIMIZATION_COMPLETE' "$W/audits/round36_optimization_window/window.log" 2>/dev/null; do
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
for batch in 1 2 4; do
  printf 'RNAFORMER_START batch=%s time=%s\n' "$batch" "$(date -Is)" | tee -a "$AUDIT/window.log"
  "$PY" tools/speed_end_to_end_4090.py --model rnaformer --batch-size "$batch" --repeats 3 \
    --out "$OUT/rnaformer_b${batch}_full.json" >> "$AUDIT/rnaformer.log" 2>&1
  printf 'RNAFORMER_PASS batch=%s time=%s\n' "$batch" "$(date -Is)" | tee -a "$AUDIT/window.log"
done
printf 'BASELINE_COMPLETE time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
