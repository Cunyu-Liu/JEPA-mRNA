#!/usr/bin/env bash
set -euo pipefail
REPO=/home/rna_jev_xens2_dualview_round56
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
cd "$REPO"
while [ ! -f "$ROOT/label_qa_summary.json" ]; do sleep 20; done
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
qa = json.loads((root / "label_qa_summary.json").read_text())
assert qa["status"] == "PASS_INDEPENDENT_INTERFACE_LABEL_QA"
assert qa["splits"]["train"]["n_valid"] == 804
assert qa["splits"]["holdout"]["n_valid"] == 202
assert not qa["train_holdout_group_overlap"] and not qa["train_holdout_pdb_overlap"]
assert qa["train_holdout_exact_sequence_overlap"] == 0
PY
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for MODE in h0 h2 rinalmo; do
  for SPLIT in train holdout; do
    "$PY" tools/extract_interface_features.py --mode "$MODE" --split "$SPLIT" >> "$ROOT/features_${MODE}_${SPLIT}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 10; done
  done
done
wait
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
for mode in ("h0", "h2", "rinalmo"):
    for split in ("train", "holdout"):
        result = json.loads((root / "features" / f"{mode}_{split}_part0.json").read_text())
        assert result["status"] == "PASS_FROZEN_INTERFACE_FEATURES"
        print(mode, split, result["n_total"], result["peak_cuda_memory_bytes"])
PY
