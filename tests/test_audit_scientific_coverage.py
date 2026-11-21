from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_scientific_coverage import (baseline_coverage, cap_pair_stats, dataset_distribution,
                                       frozen_split_consistency, leakage_audit, summarize_dataset)


def test_leakage_layers_are_independent():
    train = [{"name": "a", "seq": "AUGC", "family": "F1"}, {"name": "b", "seq": "GGGG", "family": "F2"}]
    test = [{"name": "a", "seq": "CCCC", "family": "F3"}]
    result = leakage_audit(train, test, family_available=True)
    assert result["name"]["status"] == "FAIL_NAME_OVERLAP"
    assert result["sequence"]["status"] == "PASS_EXACT_SEQUENCE_CHECK"
    assert result["family"]["status"] == "PASS_FAMILY_ID_CHECK"
    assert result["near_duplicate"]["status"].startswith("BLOCKED")
    test[0]["seq"] = "AUGC"
    test[0]["family"] = "F1"
    result = leakage_audit(train, test, family_available=True)
    assert result["sequence"]["status"] == "FAIL_EXACT_SEQUENCE_OVERLAP"
    assert result["family"]["status"] == "FAMILY_OVERLAP_PRESENT"


def test_cap_pair_stats_reports_truncation_ceiling():
    records = [{"seq": "GGGAAAACCC", "pairs": [[0, 9], [1, 8], [2, 7]]}]
    full = cap_pair_stats(records, 10)
    assert full["n_pairs_original"] == 3
    assert full["n_pairs_retained"] == 3
    assert full["model_mask_max_recall"] == 1.0
    assert full["task_truncation_max_recall"] == 1.0
    truncated = cap_pair_stats(records, 8)
    assert truncated["n_dropped_truncation"] == 2
    assert truncated["n_pairs_retained"] == 1
    assert truncated["model_mask_max_recall"] == pytest.approx(1 / 3)
    assert truncated["task_truncation_max_recall"] == pytest.approx(1 / 3)


def test_rnaformer_variants_are_not_collapsed(tmp_path):
    for suffix in ("bio", "interfam"):
        (tmp_path / f"baselines_rnaformer_{suffix}_ref_pdb_ts3.json").write_text(json.dumps({
            "n_sequences": 2, "baselines": {"rnaformer": {"n_sequences": 2}},
        }))
    result = baseline_coverage(tmp_path, "ref_pdb_ts3", 2)
    entries = [entry for entry in result["entries"] if entry["method"] == "rnaformer"]
    assert len(entries) == 2
    assert all(entry["status"] == "BLOCKED_MODEL_IDENTITY_MULTIPLE_VARIANTS" for entry in entries)
    assert result["unified_panel_complete"] is False


def test_manifest_hash_is_checked(tmp_path):
    data = tmp_path / "x.jsonl"
    data.write_text('{"name":"x","seq":"AUGC","pairs":[]}\n')
    from audit_scientific_coverage import sha256_file
    (tmp_path / "x.manifest.json").write_text(json.dumps({"out_sha256": sha256_file(data)}))
    records = [{"name": "x", "seq": "AUGC", "pairs": []}]
    assert summarize_dataset("x", data, records)["provenance_status"] == "PASS"


def test_distribution_counts_stem_and_loop_topology():
    result = dataset_distribution("x", [{"seq": "GGGAAAACCC", "pairs": [[0, 9], [1, 8], [2, 7]]}])
    assert result["n_helices"] == 1
    assert result["helix_length_histogram"] == {"3": 1}
    assert result["loop_type_counts"] == {"STACK": 2, "HAIRPIN": 1}
    assert result["n_records_with_multiloop"] == 0
    assert result["pair_span_histogram"]["4-10"] == 3
    assert sum(result["pair_span_histogram"].values()) == 3


def test_frozen_registry_compares_unique_sequences_not_record_count(tmp_path):
    frozen = tmp_path / "eval_splits_frozen.json"
    frozen.write_text(json.dumps({"splits": {"x": {"n_unique": 2, "sha256": "abc"}}}))
    summaries = {"x": {"n_unique_sequences": 2, "n_sequences": 3, "sha256": "abc"}}
    assert frozen_split_consistency(frozen, summaries)["splits"]["x"]["status"] == "MATCH"
    summaries["x"]["n_unique_sequences"] = 1
    assert frozen_split_consistency(frozen, summaries)["splits"]["x"]["status"] == "MISMATCH"
