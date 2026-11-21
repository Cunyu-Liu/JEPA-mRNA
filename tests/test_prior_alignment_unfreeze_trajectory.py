import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from analyze_prior_alignment_unfreeze_trajectory import ARMS, METRICS, aggregate_trajectory, collect_records


def test_collect_records_uses_terminal_export_layout(tmp_path):
    for seed in (0, 1, 2):
        stage = tmp_path / f"full_seed{seed}"
        raw_dir = stage / "export_train_on/bprna_vl0"
        raw_dir.mkdir(parents=True)
        np.savez(raw_dir / "abc.npz", seq="GCAUGC")
        for step in (3200, 9600, 16800, 20000):
            root = stage / "decode_vl0" if step == 20000 else stage / "trajectory" / f"step{step}"
            root.mkdir(parents=True)
            summary = {
                "seed": seed,
                "old_external_panels_used": False,
                "arms": {arm: {metric: 0.0 for metric in METRICS} for arm in ARMS},
            }
            (root / "vl0_selected_summary.json").write_text(json.dumps(summary))
            rows = []
            for arm in ARMS:
                row = {
                    "sequence_sha256": "abc",
                    "name": "example",
                    "arm": arm,
                    "pair": {"tp": 0, "fp": 0, "fn": 0},
                    "stem": {"tp": 0, "fp": 0, "fn": 0},
                }
                if step != 20000:
                    row["length"] = 6
                    row["length_bucket"] = "001-064"
                rows.append(row)
            (root / "vl0_selected_per_sequence.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

    records = collect_records(tmp_path)
    assert len(records) == 12
    assert all(row["length"] == 6 for record in records for row in record["rows"])
    result = aggregate_trajectory(records)
    assert result["status"] == "PASS"
    assert result["steps"]["20000"]["stem_length_buckets"]["001-064"]["n_rows"] == 12
