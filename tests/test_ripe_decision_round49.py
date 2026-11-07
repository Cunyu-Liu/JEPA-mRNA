import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.adjudicate_ripe_round48 import select_next_base
from tools.compare_ripe_paired_round48 import SPLITS, common_rows, subset


def result(names, raw_n, unsupported):
    rows = []
    for name, length in names:
        rows.append({"name": name, "length": length, "tp": 1, "fp": 0, "fn": 1,
                     "span_30": {"tp": 1, "fp": 0, "fn": 1},
                     "span_100": {"tp": 1, "fp": 0, "fn": 1}})
    return {"data_sha256": "same", "raw_n": raw_n, "n": len(rows),
            "n_long_unsupported": unsupported, "per_sequence": rows,
            "checkpoint_sha256": "checkpoint"}


def test_common_support_never_imputes_right_only_sequences():
    left = result([("a", 300)], 2, 1)
    right = result([("a", 300), ("b", 600)], 2, 0)
    left_rows, right_rows, extra = common_rows(left, right)
    assert [row["name"] for row in right_rows] == ["a"] and extra == ["b"]
    assert subset(left_rows, "length_200_400")[0]["name"] == "a"
    with pytest.raises(AssertionError):
        common_rows(right, left)


def test_next_base_includes_r0_and_ignores_test_metrics():
    chosen = select_next_base([
        {"experiment": "R0", "mean_best_vl0": 18.8, "test_f1": 0.0},
        {"experiment": "ripe_r3", "mean_best_vl0": 18.9, "test_f1": 1.0}])
    assert chosen["experiment"] == "R0"


def test_full_paired_comparison_with_extra_encoder_support(tmp_path):
    root = tmp_path
    manifest = {"selected": {"experiment": "structrfm_s3", "run_prefix": "structrfm_s3_pairmlp490_s"},
                "arms": [{"arm": "ripe_encoder_r3", "mode": "r3", "encoder": True}]}
    path = root / "audits/round48/ripe_preregistered.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    for experiment in ("structrfm_s3", "ripe_encoder_r3"):
        for seed in range(3):
            for split in SPLITS:
                if split == "archiveii":
                    names = [("a", 300)] if experiment == "structrfm_s3" else [("a", 300), ("b", 600)]
                    value = result(names, 2, int(experiment == "structrfm_s3"))
                else:
                    value = result([("a", 300)], 1, 0)
                out = root / "eval_decision" / experiment / f"s{seed}" / f"{split}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(value))
    out = root / "ripe_paired.json"
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / "tools/compare_ripe_paired_round48.py"),
                    "--root", str(root), "--out", str(out), "--bootstrap", "1000"], check=True)
    value = json.loads(out.read_text())
    rows = value["comparisons"]["ripe_encoder_r3"]
    archive = next(row for row in rows if row["split"] == "archiveii")
    assert archive["right_only_supported_n"] == 1 and archive["n"] == 1
    assert archive["left_micro_f1"] == archive["right_micro_f1"]
    assert len(value["long_subsets"]["ripe_encoder_r3"]) == 24
