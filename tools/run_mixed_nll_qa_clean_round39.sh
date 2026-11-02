#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
run=$root/eval_decision/round39_mixed_nll/runs/nll_s0
clean_root=$root/eval_decision/round39_clean_benchmark/clean_splits/tr1c
"$python" - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); meta=json.loads((run/"run_meta.json").read_text())
assert meta["status"]=="completed" and meta["converged"] and meta["early_stopped"]
assert meta["objective"]["lambda_nll"]==1 and meta["objective"]["lambda_distill"]==0
assert (run/"best.pt").is_file() and (run/"resume.pt").is_file()
print("NLL_CONVERGENCE_CONFIRMED_FOR_QA",meta["steps_completed"],meta["best_step"],flush=True)
PY
"$python" tools/qa_plana_student_round39.py \
    --run-dir "$run" \
    --data /root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl \
    --teacher-dir "$root/eval_decision/round39_xens2/teacher_vl0" \
    --out "$root/eval_decision/round39_mixed_nll/qa/nll_s0_vl0" \
    > logs/round39_mixed_nll/nll_s0_qa.log 2>&1
for split in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
    out=$root/eval_decision/round39_mixed_nll/nll_s0/clean/$split.json
    if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert (x["status"]=="PASS" and len(x["runs"])==3 or x["status"]=="EMPTY_AFTER_DELEAK" and not x["runs"])
PY
    then
        continue
    fi
    "$python" tools/speed_end_to_end_4090.py --model plana \
        --plana-weights "$run/best.pt" --batch-size 1 --repeats 3 --allow-long \
        --data "$clean_root/$split.jsonl" --clean-audit "$clean_root/$split.audit.json" --out "$out" \
        > "logs/round39_mixed_nll/nll_s0_clean_${split}.log" 2>&1
done
echo NLL_SEED0_QA_AND_CLEAN_8_SPLIT_PASS
