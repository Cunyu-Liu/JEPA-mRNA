from __future__ import annotations

import hashlib

from tools.audit_foldnar_label_quality_v1 import (
    DATASETS,
    STATES,
    conservation,
    issue_reasons,
    make_record,
    stable_id,
)
from tools.verify_foldnar_label_quality_v1 import bracket_pairs, crossing_members, recompute


def metric_fixture(raw, dataset="chanrg_train", index=0):
    calculated = recompute(raw)
    qa = dict(calculated["qa"])
    sequence = calculated["sequence"]
    return {
        "dataset": dataset,
        "source_row_index": index,
        "id": raw.get("name", raw.get("id", "")),
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "length": len(sequence),
        "label_source": DATASETS[dataset]["source_category"],
        "source": raw.get("source", "CHANRG_Rfam15_consensus"),
        "family": raw.get("family"),
        "clan": raw.get("clan"),
        "genomic_parent": None,
        "structure_pair_symmetry": calculated["observation"],
        "qa": qa,
        "n_pairs": qa["unique_normalized_pairs"],
        "n_valid_pairs": calculated["one_to_one"],
    }


def test_zero_based_coordinates_and_pair_conservation():
    raw = {"id": "x", "seq": "AUGCAU", "structure": "(....)", "pairs": [[5, 0], [0, 5]]}
    record = make_record("chanrg_train", "chanrg_train", 0, raw, metric_fixture(raw))
    assert record["coordinate_convention"].startswith("0-based inclusive")
    assert record["raw_pair_entries"] == 2
    assert record["unique_normalized_pairs"] == 1
    assert record["duplicate_pair_entries"] == 1
    assert record["review_detection_status"] == "EXACT"
    assert record["proposed_exclusion_candidate"] is True
    assert record["actual_deleted"] is False


def test_crossing_is_independently_detected_without_silent_repair():
    pairs = [(0, 5), (2, 7), (9, 10)]
    assert crossing_members(pairs) == {(0, 5), (2, 7)}
    raw = {"id": "x", "seq": "AUGCAUGC", "structure": "........", "pairs": [list(pair) for pair in pairs[:2]]}
    result = recompute(raw)
    assert result["qa"]["crossing_pairs"] == 2
    assert result["qa"]["raw_pair_entries"] == 2


def test_canonical_min_loop_partner_conflict_and_invalid_coordinates():
    raw = {
        "id": "x", "seq": "AAAAAA", "structure": "......",
        "pairs": [[0, 2], [0, 5], [1, 4], [-1, 3], [4, 4], [2]],
    }
    result = recompute(raw)
    assert result["qa"]["partner_conflict_nucleotides"] == 1
    assert result["qa"]["out_of_range_pairs"] == 1
    assert result["qa"]["self_pairs"] == 1
    assert result["qa"]["malformed_pair_entries"] == 1
    assert result["qa"]["min_loop_span_lt4_pairs"] == 1
    assert result["qa"]["noncanonical_pairs"] == 1


def test_symmetry_parser_rejects_non_dot_bracket_tokens():
    assert bracket_pairs("(..)") == {(0, 3)}
    try:
        bracket_pairs("[..]")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported brackets must not be silently accepted")


def test_no_flag_is_unknown_not_clean_and_no_deletion():
    raw = {"id": "x", "seq": "AUGCAU", "structure": "(....)", "pairs": [[0, 5]]}
    record = make_record("chanrg_train", "chanrg_train", 0, raw, metric_fixture(raw))
    assert record["review_reasons"] == []
    assert record["review_detection_status"] == "NOT_DETECTED_UNKNOWN"
    assert record["review_detection_status"] != "CLEAN"
    account = conservation(1, 0, 0)
    assert account["conservation_holds"] is True
    assert account["actual_deleted_records"] == 0
    assert account["deletion_authorized"] is False


def test_source_categories_do_not_upgrade_comparative_or_database_labels():
    assert DATASETS["tr1c"]["source_category"] == "DATABASE"
    assert DATASETS["chanrg_train"]["source_category"] == "COMPARATIVE_INFERRED"
    assert DATASETS["chanrg_validation"]["source_category"] == "COMPARATIVE_INFERRED"
    raw = {"id": "AAAA_1-6", "seq": "AUGCAU", "structure": "(....)", "pairs": [[0, 5]]}
    record = make_record("chanrg_validation", "chanrg_validation", 3, raw, metric_fixture(raw, "chanrg_validation", 3))
    assert record["source_category"] != "EXPERIMENTAL"
    assert record["experimental_lineage_status"] == "UNCHECKED"
    assert record["modified_parent_status"] == "UNCHECKED"
    assert record["pdb_mmcif_status"] == "UNCHECKED"
    assert record["pdb_mmcif_checked"] is False


def test_declared_metadata_conflicts_are_accounted_not_repaired():
    raw = {
        "id": "x", "seq": "AUGCAU", "structure": "(....)", "pairs": [[0, 5]],
        "n_pairs": 2, "n_crossing": 1, "is_pseudoknot": True,
    }
    metric = metric_fixture(raw)
    hard, soft = issue_reasons(metric, raw)
    assert "declared_n_pairs_mismatch" in hard
    assert "declared_n_crossing_mismatch" in hard
    assert "declared_pseudoknot_mismatch" in hard
    assert soft == []


def test_stable_id_preserves_source_row_and_allowed_status_vocabulary():
    assert stable_id("tr1c", 7, "plk#2") == "bprna_tr1c:row000007:plk#2"
    assert stable_id("chanrg_train", 7, "x") == "chanrg_train:row000007:x"
    assert STATES == {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
