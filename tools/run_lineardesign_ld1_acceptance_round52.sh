#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?matrix or controls}
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
REPO=/home/rna_jev
PY=/root/miniconda3/envs/codonflow/bin/python
export PYTHONPATH="$REPO:$REPO/src:$REPO/tools:$REPO/eval"
case "$MODE" in
  matrix)
    exec 9>/tmp/round52_ld1_matrix_acceptance.lock
    flock -n 9 || { echo "another LD1 matrix acceptance is active"; exit 1; }
    until test -f "$ROOT/runs/lineardesign_ld1/manifest.json"; do sleep 300; done
    "$PY" "$REPO/tools/qa_lineardesign_ld1_round52.py" \
      --fasta "$ROOT/data/lineardesign_round52/uniprot_114.fasta" \
      --split "$ROOT/audits/round52/lineardesign_ld1_design_split.json" \
      --matrix-dir "$ROOT/runs/lineardesign_ld1" \
      --codon-table "$REPO/third_party/LinearDesign/codon_usage_freq_table_human.csv" \
      --output "$ROOT/audits/round52/lineardesign_ld1_full_qa.json" \
      > "$ROOT/logs/round52/lineardesign_ld1_full_qa.log" 2>&1
    for run in 1 2 3; do
      "$PY" "$REPO/tools/test_aggregate_lineardesign_ld1_matrix_round52.py" || exit 1
    done
    "$PY" "$REPO/tools/aggregate_lineardesign_ld1_matrix_round52.py" \
      --matrix-dir "$ROOT/runs/lineardesign_ld1" \
      --qa "$ROOT/audits/round52/lineardesign_ld1_full_qa.json" \
      --output "$ROOT/audits/round52/lineardesign_ld1_exact_beam.json" \
      > "$ROOT/logs/round52/lineardesign_ld1_exact_beam.log" 2>&1
    ;;
  controls)
    exec 9>/tmp/round52_ld1_controls_acceptance.lock
    flock -n 9 || { echo "another LD1 controls acceptance is active"; exit 1; }
    until [[ $(ls "$ROOT/runs/lineardesign_ld1_controls/cells"/*.json 2>/dev/null | wc -l) -eq 798 ]]; do sleep 300; done
    "$PY" "$REPO/tools/qa_lineardesign_control_evaluation_round52.py" \
      --controls "$ROOT/runs/lineardesign_ld1_controls/controls.json" \
      --fasta "$ROOT/data/lineardesign_round52/uniprot_114.fasta" \
      --codon-table "$REPO/third_party/LinearDesign/codon_usage_freq_table_human.csv" \
      --cells-dir "$ROOT/runs/lineardesign_ld1_controls/cells" \
      --output "$ROOT/audits/round52/lineardesign_ld1_controls_qa.json" \
      > "$ROOT/logs/round52/lineardesign_ld1_controls_qa.log" 2>&1
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2;;
esac
"$PY" - "$ROOT" "$MODE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
if mode == "matrix":
    paths = [root / "audits/round52/lineardesign_ld1_full_qa.json",
             root / "audits/round52/lineardesign_ld1_exact_beam.json"]
    reports = [json.loads(path.read_text()) for path in paths]
    assert reports[0]["complete"] and reports[0]["n"] == 456
    assert reports[1]["status"] == "PASS_LD1_EXACT_BEAM_COMPUTATIONAL_COMPARISON"
else:
    paths = [root / "audits/round52/lineardesign_ld1_controls_qa.json"]
    reports = [json.loads(paths[0].read_text())]
    assert reports[0]["n_recomputed"] == 798
result = {"status": f"PASS_LD1_{mode.upper()}_THREE_CHECK_ACCEPTANCE",
          "scientific_conclusion": False,
          "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
          "limitations": ["Full computational QA is complete; wild-type assets and LD2 frozen-model evaluation remain separate.",
                          "No biological efficacy or PRISM benefit is claimed."]}
out = root / "audits/round52" / f"lineardesign_ld1_{mode}_acceptance.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
