#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/pylibs:$REPO:$REPO/src:$REPO/tools:$REPO/eval
while ! $PY -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["allow_lineardesign"]' "$ROOT/audits/round48/complex_d1_decision.json" 2>/dev/null; do sleep 60; done
$PY "$REPO/tools/prepare_lineardesign_round49.py" --root "$ROOT" \
  --lineardesign "$REPO/third_party/LinearDesign" \
  --out-dir "$ROOT/eval_decision/lineardesign_round49" > "$ROOT/logs/round48/lineardesign_prepare.log" 2>&1
readarray -t C < <($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(m["checkpoint"],m["checkpoint_sha256"],m["best_step"]) for m in d["models"]]' "$ROOT/eval_decision/lineardesign_round49/manifest.json")
for index in 0 1; do
  offset=$((index * 3))
  checkpoint=${C[$offset]}
  checkpoint_sha=${C[$((offset + 1))]}
  best_step=${C[$((offset + 2))]}
  if [ "$index" -eq 0 ]; then experiment=complex_d0; else experiment=complex_d1; fi
  out=$ROOT/eval_decision/lineardesign_round49/${experiment}.json
  if [ -f "$out" ] && $PY -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS" and d["device"]=="cuda" and d["raw_n"]==d["n"]==3 and d["repeats"]==3; assert d["checkpoint_sha256"]==sys.argv[2] and d["checkpoint_step"]==int(sys.argv[3]) and d["score_forward_count_per_sequence"]==1' "$out" "$checkpoint_sha" "$best_step" 2>/dev/null; then continue; fi
  $PY "$REPO/tools/eval_structrfm_full_round48.py" --checkpoint "$checkpoint" \
    --pretrained "$ROOT/weights/structrfm_round48" \
    --data "$ROOT/eval_decision/lineardesign_round49/candidates.jsonl" \
    --out "$out" --repeats 3 > "$ROOT/logs/round48/lineardesign_${experiment}.log" 2>&1
done
$PY "$REPO/tools/summarize_lineardesign_round49.py" --root "$ROOT" \
  --out "$ROOT/eval_decision/lineardesign_round49/summary.json" > "$ROOT/logs/round48/lineardesign_summary.log" 2>&1
$PY "$REPO/tools/summarize_lineardesign_round49.py" --root "$ROOT" \
  --out "$ROOT/eval_decision/lineardesign_round49/summary.recheck.json" >> "$ROOT/logs/round48/lineardesign_summary.log" 2>&1
cmp "$ROOT/eval_decision/lineardesign_round49/summary.json" "$ROOT/eval_decision/lineardesign_round49/summary.recheck.json"
