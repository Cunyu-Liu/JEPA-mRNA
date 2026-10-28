#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/lucaone/bin/python
TOOLS=/home/rna_jev/tools
BASE=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision

run_qa() {
  local name="$1"
  shift
  echo "OTHER_FULL_QA_START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@"
  echo "OTHER_FULL_QA_PASS $name $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_qa nb_splice_acceptor "$PY" "$TOOLS/qa_other_full_round34.py" --kind nb_splice --ss-type acceptor --result "$BASE/nb_splice_acceptor_round33_full_v1/result.json"
run_qa nb_splice_donor "$PY" "$TOOLS/qa_other_full_round34.py" --kind nb_splice --ss-type donor --result "$BASE/nb_splice_donor_round33_full_v1/result.json"
run_qa rinalmo_splice_acceptor "$PY" "$TOOLS/qa_other_full_round34.py" --kind rinalmo_splice --ss-type acceptor --result "$BASE/rinalmo_splice_acceptor_round33_full_v1/result.json"
run_qa rinalmo_splice_donor "$PY" "$TOOLS/qa_other_full_round34.py" --kind rinalmo_splice --ss-type donor --result "$BASE/rinalmo_splice_donor_round33_full_v1/result.json"
run_qa rinalmo_ncrna_clean "$PY" "$TOOLS/qa_ncrna_full.py" --noise clean
run_qa rinalmo_ncrna_bn200 "$PY" "$TOOLS/qa_ncrna_full.py" --noise bn200

echo "ALL_OTHER_FULL_QA_PASS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
