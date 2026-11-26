from __future__ import annotations

import math

from tools.build_foldnar_model_dev_registry_v3 import (
    EXPECTED_COUNTS,
    EXPECTED_PAIRS,
    FINAL_COLUMNS,
    MINIMUM_SAMPLE,
    SENSITIVITY_COLUMNS,
    concentration,
    final_exposure_view,
    minimum_sample_gate,
    public_summary,
)


def row(family, pairs=10):
    return {"family": family, "one_to_one_valid_pairs": pairs}


def test_concentration_uses_all_weight_for_top_and_mapped_weight_for_hhi():
    result = concentration([row("A", 5), row("A", 3), row("B", 4), row(None, 7)])
    assert result["n_sequences"] == 4
    assert result["n_pairs"] == 19
    assert result["n_known_families"] == 2
    assert result["top_known_family_by_sequences"] == "A"
    assert result["top_known_family_sequence_fraction_of_all"] == 0.5
    assert result["top_known_family_pair_fraction_of_all"] == 8 / 19
    assert math.isclose(result["known_family_sequence_hhi"], (2 / 3) ** 2 + (1 / 3) ** 2)
    assert math.isclose(result["known_family_pair_hhi"], (8 / 12) ** 2 + (4 / 12) ** 2)
    assert result["unmapped_or_ambiguous_sequence_fraction"] == 0.25
    assert result["unmapped_or_ambiguous_pair_fraction"] == 7 / 19


def test_original_minimum_sample_thresholds_pass_at_exact_boundary():
    rows = [row(f"RF{index:05d}", 10) for index in range(10) for _ in range(10)]
    stats = concentration(rows)
    assert stats["n_sequences"] == 100
    assert stats["n_pairs"] == 1000
    assert stats["n_known_families"] == 10
    assert minimum_sample_gate(stats)["pass"] is True
    assert MINIMUM_SAMPLE == {
        "min_sequences": 100,
        "min_pairs": 1000,
        "min_known_families": 10,
        "max_top_family_fraction": 0.5,
        "max_family_hhi": 0.25,
        "max_unmapped_family_fraction": 0.5,
    }


def test_below_sequence_and_pair_minima_fails_without_changing_thresholds():
    rows = [row(f"RF{index:05d}", 9) for index in range(10) for _ in range(9)]
    gate = minimum_sample_gate(concentration(rows))
    assert gate["pass"] is False
    assert set(gate["failed_gates"]) == {"min_sequences", "min_pairs"}
    assert gate["thresholds"] == MINIMUM_SAMPLE


def test_unmapped_stratum_is_diagnostic_even_with_many_sequences_and_pairs():
    stats = concentration([row(None, 33) for _ in range(626)])
    gate = minimum_sample_gate(stats)
    assert stats["n_known_families"] == 0
    assert stats["known_family_sequence_hhi"] is None
    assert stats["unmapped_or_ambiguous_sequence_fraction"] == 1.0
    assert gate["pass"] is False
    assert set(gate["failed_gates"]) == {
        "min_known_families", "max_family_hhi", "max_unmapped_family_fraction",
    }


def test_family_macro_component_minimum_is_not_silently_dropped():
    stats = concentration([row("A", 250), row("A", 250)] + [row("B", 500) for _ in range(30)])
    assert stats["n_macro_component_eligible_families"] == 1
    assert stats["family_counts"]["A"] == {"n_sequences": 2, "n_pairs": 500}


def test_frozen_population_conservation():
    assert sum(EXPECTED_COUNTS["tr1c_cluster80_dev"].values()) == 9143
    assert sum(EXPECTED_PAIRS["tr1c_cluster80_dev"].values()) == 247013
    assert sum(EXPECTED_COUNTS["tr1c_cluster80_train"].values()) == 33421
    assert sum(EXPECTED_PAIRS["tr1c_cluster80_train"].values()) == 910260
    assert EXPECTED_COUNTS["tr1c_cluster80_dev"] == {
        "EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626, "UNCHECKED": 0,
    }


def test_bprna_new_is_final_only_and_not_clean_ood():
    summary = {
        "populations": {
            "bprna_new": {
                "n_input_rows": 5388,
                "known_source_exact_or_homology_state": {
                    "EXACT": 5381, "HOMOLOGY": 7, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 0,
                },
                "complete_official_pretraining_exposure_state": {
                    "EXACT": 5381, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 7,
                },
            },
        },
    }
    result = final_exposure_view(summary)
    assert result["n_sequences"] == 5388
    assert result["selection_use"] is False
    assert result["clean_ood_or_pretraining_unseen_claim_allowed"] is False
    assert result["complete_official_pretraining_manifest"] == "UNCHECKED"
    assert result["unknown_is_not_clean"] is True


def test_report_columns_are_additive_and_cannot_select_route():
    assert "exposure_paired_ci95" in SENSITIVITY_COLUMNS
    assert "route_selection_allowed" in SENSITIVITY_COLUMNS
    assert "final_exposure_metric" in FINAL_COLUMNS
    assert "selection_use" in FINAL_COLUMNS
    assert "family_counts" not in public_summary({"family_counts": {"A": {}}, "n_pairs": 2})
