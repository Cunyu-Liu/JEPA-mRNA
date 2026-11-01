#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
BASE=$W/eval_decision/speed_round36
OUT=$W/eval_decision/speed_round37
AUDIT=$W/audits/round37_speed_window
mkdir -p "$OUT" "$AUDIT"
printf 'START time=%s\n' "$(date -Is)" | tee "$AUDIT/window.log"
if ps -eo args= | grep -E '[n]b_secstr_dual_round34.py|[q]a_nb_secstr_round34.py|[r]un_nb_secstr_queue_round35.py' > /dev/null; then
  echo 'NB process detected; refusing non-exclusive timing' >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU compute process detected; refusing non-exclusive timing' >&2
  exit 1
fi
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
  local model=$1 batch=$2 baseline=$3 extra=$4
  printf 'CANDIDATE model=%s batch=%s baseline=%s extra=%s time=%s\n' "$model" "$batch" "$baseline" "$extra" "$(date -Is)" | tee -a "$AUDIT/window.log"
  "$PY" tools/profile_speed_round36.py --model "$model" --batch-size "$batch" --repeats 3 \
    --baseline "$BASE/$baseline" $extra \
    --out "$OUT/${model}_b${batch}${extra// /_}_profile.json" >> "$AUDIT/candidates.log" 2>&1
}
run_candidate xens2 16 profile/xens2_b4_profile.json ''
run_candidate r2d 32 profile/r2d_b4_profile.json ''
run_candidate plana 32 profile/plana_b4_profile.json ''
run_candidate xens2 32 profile/xens2_b4_profile.json ''
run_candidate r2d 16 optimized/r2d_b16_profile.json '--tf32'
printf 'ROUND37_SPEED_WINDOW_COMPLETE time=%s\n' "$(date -Is)" | tee -a "$AUDIT/window.log"
nvidia-smi > "$AUDIT/nvidia_smi_final.txt"
