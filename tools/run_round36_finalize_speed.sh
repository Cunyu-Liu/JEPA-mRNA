#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
W=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
ROOT=$W/eval_decision/speed_round36
OUT=$W/figure_assets/round36
AUDIT=$W/audits/round36_finalize_speed
mkdir -p "$OUT" "$AUDIT"
printf 'WAIT_BASELINE time=%s\n' "$(date -Is)" | tee "$AUDIT/finalize.log"
while ! grep -q 'BASELINE_COMPLETE' "$W/audits/round36_baseline_window/window.log" 2>/dev/null; do
  sleep 60
done
"$PY" tools/build_round36_speed_comparison.py \
  --profile-dir "$ROOT/profile" \
  --optimized-dir "$ROOT/optimized" \
  --baseline-dir "$ROOT/baselines" \
  --out "$ROOT/speed_round36_summary.json" > "$AUDIT/summary.log"
export SPEED_4090_ROUND33_DIR=$W/eval_decision/speed_4090_round33
export SPEED_4090_ROUND34_DIR=$W/eval_decision/speed_4090_round34
"$PY" tools/make_figure5_round36.py --summary "$ROOT/speed_round36_summary.json" --out-dir "$OUT" > "$AUDIT/figure.log"
"$PY" - <<'PY'
import json
from pathlib import Path
from PIL import Image
root=Path('/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024')
summary=json.loads((root/'eval_decision/speed_round36/speed_round36_summary.json').read_text())
assert summary['status']=='PASS'
assert summary['reference_models']
source=json.loads((root/'figure_assets/round36/fig05_accuracy_speed_4090_round36_source.json').read_text())
assert source['cross_batch_numerical_equivalence']
with Image.open(root/'figure_assets/round36/fig05_accuracy_speed_4090_round36_3840x1920.png') as im:
    assert im.size==(3840,1920)
print('ROUND36_SPEED_ARTIFACT_QA_PASS',len(summary['best_eligible_by_model']),len(summary['reference_models']))
PY
printf 'FINALIZE_COMPLETE time=%s\n' "$(date -Is)" | tee -a "$AUDIT/finalize.log"
