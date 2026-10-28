#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/lucaone/bin/python
TOOLS=/home/rna_jev/tools
BASE=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision
ORIGINAL="$TOOLS/rinalmo_splice_dual_round33_dee8879.py"

for ss_type in acceptor donor; do
  echo "RINALMO_SPLICE_RECOVERY_START $ss_type $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" "$TOOLS/rebuild_rinalmo_splice_cache_round34.py" --ss-type "$ss_type" --batch-size 32
  "$PY" "$TOOLS/qa_other_full_round34.py" --kind rinalmo_splice --ss-type "$ss_type" --code-override "$ORIGINAL" --result "$BASE/rinalmo_splice_${ss_type}_round33_full_v1/result.json"
  echo "RINALMO_SPLICE_RECOVERY_QA_PASS $ss_type $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "RINALMO_SPLICE_CACHE_AND_QA_PASS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
