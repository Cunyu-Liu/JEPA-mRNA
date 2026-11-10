#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/lucaone/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>/tmp/round52_rinalmo_replay.lock
flock -n 9 || { echo "another RiNALMo replay is active"; exit 1; }
if [[ ! -f "$ROOT/audits/round52/rinalmo_r0_training_qa.json" ]] || \
   [[ ! -f "$ROOT/audits/round52/rinalmo_r0_training_qa.recheck.json" ]] || \
   ! cmp -s "$ROOT/audits/round52/rinalmo_r0_training_qa.json" \
            "$ROOT/audits/round52/rinalmo_r0_training_qa.recheck.json"; then
  "$PY" "$REPO/tools/qa_rinalmo_r0_training_round52.py" \
    --root "$ROOT" --output "$ROOT/audits/round52/rinalmo_r0_training_qa.json" \
    > "$ROOT/logs/round52/rinalmo_r0_training_qa.log" 2>&1
  "$PY" "$REPO/tools/qa_rinalmo_r0_training_round52.py" \
    --root "$ROOT" --output "$ROOT/audits/round52/rinalmo_r0_training_qa.recheck.json" \
    > "$ROOT/logs/round52/rinalmo_r0_training_qa.recheck.log" 2>&1
fi
cmp "$ROOT/audits/round52/rinalmo_r0_training_qa.json" \
    "$ROOT/audits/round52/rinalmo_r0_training_qa.recheck.json"
"$PY" - "$ROOT/audits/round52/rinalmo_r0_training_qa.json" <<'PY'
import json
import sys
qa = json.load(open(sys.argv[1]))
assert qa["status"] == "PASS_R0_TRAINING_THREE_CHECK_QA" and qa["allow_r_replay"]
assert set(qa["runs"]) == {"42", "2026", "3407"}
PY
for seed in 42 2026; do
  "$PY" "$REPO/tools/replay_r6_objective_optimizer_round51.py" \
    --backbone rinalmo \
    --run-dir "$ROOT/runs/round52_rinalmo_r0_s${seed}" \
    --pretrained /root/rna_jepa_data/weights/rinalmo-giga \
    --output "$ROOT/audits/round52/rinalmo_replay_s${seed}.json" \
    >> "$ROOT/logs/round52/rinalmo_replay_s${seed}.log" 2>&1
done
for seed in 42 2026; do
  expected=$("$PY" - "$ROOT/audits/round52/rinalmo_r0_training_qa.json" "$seed" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["runs"][sys.argv[2]]["n_unfrozen"])
PY
)
  "$PY" "$REPO/tools/qa_r6_replay_round51.py" \
    --backbone rinalmo --expected-unfrozen "$expected" \
    --result "$ROOT/audits/round52/rinalmo_replay_s${seed}.json" \
    --output "$ROOT/audits/round52/rinalmo_replay_s${seed}.qa.json" \
    > "$ROOT/logs/round52/rinalmo_replay_s${seed}.qa.log" 2>&1
  "$PY" "$REPO/tools/qa_r6_replay_round51.py" \
    --backbone rinalmo --expected-unfrozen "$expected" \
    --result "$ROOT/audits/round52/rinalmo_replay_s${seed}.json" \
    --output "$ROOT/audits/round52/rinalmo_replay_s${seed}.qa.recheck.json" \
    > "$ROOT/logs/round52/rinalmo_replay_s${seed}.qa.recheck.log" 2>&1
  cmp "$ROOT/audits/round52/rinalmo_replay_s${seed}.qa.json" \
      "$ROOT/audits/round52/rinalmo_replay_s${seed}.qa.recheck.json"
done
"$PY" "$REPO/tools/decide_rinalmo_replay_round52.py" \
  --result "$ROOT/audits/round52/rinalmo_replay_s42.json" \
  --result "$ROOT/audits/round52/rinalmo_replay_s2026.json" \
  --qa "$ROOT/audits/round52/rinalmo_replay_s42.qa.json" \
  --qa "$ROOT/audits/round52/rinalmo_replay_s2026.qa.json" \
  --training-qa "$ROOT/audits/round52/rinalmo_r0_training_qa.json" \
  --output "$ROOT/audits/round52/rinalmo_replay_decision.json" \
  > "$ROOT/logs/round52/rinalmo_replay_decision.log" 2>&1
"$PY" "$REPO/tools/decide_rinalmo_replay_round52.py" \
  --result "$ROOT/audits/round52/rinalmo_replay_s42.json" \
  --result "$ROOT/audits/round52/rinalmo_replay_s2026.json" \
  --qa "$ROOT/audits/round52/rinalmo_replay_s42.qa.json" \
  --qa "$ROOT/audits/round52/rinalmo_replay_s2026.qa.json" \
  --training-qa "$ROOT/audits/round52/rinalmo_r0_training_qa.json" \
  --output "$ROOT/audits/round52/rinalmo_replay_decision.recheck.json" \
  > "$ROOT/logs/round52/rinalmo_replay_decision.recheck.log" 2>&1
cmp "$ROOT/audits/round52/rinalmo_replay_decision.json" \
    "$ROOT/audits/round52/rinalmo_replay_decision.recheck.json"
