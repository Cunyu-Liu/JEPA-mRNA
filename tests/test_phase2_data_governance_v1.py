import collections
import json

import pytest

from tools.audit_phase2_data_governance_v1 import (
    STATES,
    audit_record,
    bin_value,
    bracket_pairs,
    genomic_parent,
    numeric_summary,
    pair_features,
    summarize,
)


def test_nested_pairs_topology_loops_and_conservation():
    sequence = "GCGAAAAACGCGCGAAAAGCGC"
    pairs = bracket_pairs("(((.....)))(((....)))")
    sequence = "GCGAAAAACGCGCGAAAAGCGC"[:22]
    result = pair_features(sequence, pairs)
    assert result["qa"]["raw_pair_entries"] == len(pairs)
    assert result["qa"]["crossing_pairs"] == 0
    assert result["qa"]["partner_conflict_nucleotides"] == 0
    assert result["pair_conservation_residual"] == 0
    assert result["topology"] == "multi_root_no_multiloop"
    assert result["loops"] == {"HAIRPIN": 2}
    assert result["topology_sha256"] is not None


def test_crossing_min_loop_noncanonical_and_partner_conflict_are_not_repaired():
    sequence = "AAAAAAAAAAAA"
    result = pair_features(sequence, [[0, 5], [2, 8], [2, 9], [0, 5], [1, 2], [3, 4]])
    assert result["qa"]["duplicate_pair_entries"] == 1
    assert result["qa"]["partner_conflict_nucleotides"] >= 1
    assert result["qa"]["min_loop_span_lt4_pairs"] >= 1
    assert result["qa"]["noncanonical_pairs"] >= 1
    assert result["pair_conservation_residual"] == 0
    assert result["topology"] == "crossing_or_invalid"


def test_crossing_detection_and_pseudoknot_status():
    result = pair_features("AUGCAUGCAU", [[0, 5], [2, 8]])
    assert result["qa"]["crossing_pairs"] == 2
    assert result["qa"]["pseudoknot_records"] == 1
    assert result["topology"] == "crossing_or_invalid"


def test_multiloop_and_isolated_pair_features():
    structure = "(((.((....)).((....)).)))"
    pairs = bracket_pairs(structure)
    result = pair_features("G" * len(structure), pairs)
    assert result["topology"] == "contains_multiloop"
    assert result["loops"]["MULTILOOP"] == 1
    assert result["junctions"]
    assert result["isolated"]["stacked"] > 0


def test_structure_pair_symmetry_and_duplicate_ids_are_observable():
    raw = {"id": "AAAA02037454.1_2001-2135", "seq": "GCAU", "pairs": [[0, 3]],
           "structure": "(..)", "family": "RF00001", "clan": "CL00001"}
    record = audit_record("chanrg_train", 0, raw)
    assert record["structure_pair_symmetry"] == "MATCH"
    assert record["genomic_parent"] == "AAAA02037454.1"
    assert record["modified_parent_status"] == "UNCHECKED_NO_MMCIF_PARENT_MAPPING"
    raw["structure"] = "...."
    assert audit_record("chanrg_train", 1, raw)["structure_pair_symmetry"] == "MISMATCH"


def test_genomic_parent_parser_does_not_claim_chemical_parent():
    assert genomic_parent("AAAA02037454.1_2001-2135") == "AAAA02037454.1"
    assert genomic_parent("bpRNA_CRW_15847.bpseq") is None


def test_numeric_summary_and_boundaries():
    assert numeric_summary([1, 2, 3, 4])["median"] == 2.5
    assert numeric_summary([])["n"] == 0
    assert bin_value(3, (("le3", 0, 3), ("ge4", 4, None))) == "le3"


def test_only_allowed_leakage_states_are_defined():
    assert STATES == {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
    assert "CLEAN" not in STATES


def test_summary_keeps_duplicate_id_and_sequence_accounting():
    raw = {"id": "AAAA1_10-20", "seq": "GCAU", "pairs": [[0, 3]],
           "structure": "(..)", "family": "RF00001", "clan": None}
    first = audit_record("chanrg_train", 0, raw)
    second = audit_record("chanrg_train", 1, raw)
    result = summarize([first, second])
    assert result["n_records"] == 2
    assert result["n_duplicate_id_rows_beyond_first"] == 1
    assert result["n_duplicate_sequence_rows_beyond_first"] == 1
    assert result["macro_and_coverage_risk"]["n_undercovered_groups"]["family"] == 1


def test_unmapped_labels_are_not_macro_eligible_groups():
    raw = {"id": "row", "seq": "GCAU", "pairs": [[0, 3]], "structure": "(..)"}
    result = summarize([audit_record("chanrg_train", 0, raw)])
    risk = result["macro_and_coverage_risk"]
    assert risk["n_macro_component_eligible"] == {"family": 0, "clan": 0, "cluster80": 0}
    assert risk["n_undercovered_groups"] == {"family": 0, "clan": 0, "cluster80": 0}


def test_invalid_brackets_raise():
    with pytest.raises(ValueError):
        bracket_pairs("(..")
