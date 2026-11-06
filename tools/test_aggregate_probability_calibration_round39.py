import json

from aggregate_probability_calibration_round39 import build_cell


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_native_invalid_preserves_structure_f1_but_not_ece(tmp_path):
    split = "archiveii_embok_clean"
    write(tmp_path / "mxfold2/calibration" / f"{split}.json",
          {"status": "FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE", "n_sequences": 1041,
           "source_structure_f1_runs": [0.7, 0.7, 0.7]})
    write(tmp_path / "calibration_comparisons" / f"{split}.xens2_mxfold2.json",
          {"status": "NATIVE_BPP_INVALID_NO_CALIBRATION", "matched_n": None,
           "baseline_clean_n": 1041, "xens2_clean_n": 1107})
    cell = build_cell(tmp_path, "mxfold2", split)
    assert cell["status"] == "NATIVE_BPP_INVALID_NO_CALIBRATION"
    assert cell["own_clean_f1"] == 0.7 and cell["own_clean_n"] == 1041
    assert cell["matched_n"] is None and cell["matched_baseline"]["ece"] is None


def test_empty_clean_is_not_zero_score(tmp_path):
    split = "ref_pdb_ts3"
    write(tmp_path / "nucleicbert_ft/calibration" / f"{split}.json",
          {"status": "EMPTY_AFTER_DELEAK", "n_sequences": 0})
    write(tmp_path / "calibration_comparisons" / f"{split}.xens2_nucleicbert_ft.json",
          {"status": "EMPTY_AFTER_DELEAK", "matched_n": 0, "baseline_clean_n": 0, "xens2_clean_n": 4})
    cell = build_cell(tmp_path, "nucleicbert_ft", split)
    assert cell["status"] == "EMPTY_AFTER_DELEAK" and cell["matched_n"] == 0
    assert cell["own_clean_f1"] is None and cell["own_clean"]["ece"] is None
    (tmp_path / "calibration_comparisons" / f"{split}.xens2_nucleicbert_ft.json").unlink()
    cell = build_cell(tmp_path, "nucleicbert_ft", split)
    assert cell["status"] == "EMPTY_AFTER_DELEAK" and cell["own_clean_n"] == 0
    assert cell["own_clean_f1"] is None and cell["own_clean"]["ece"] is None


def test_vienna_metrics_and_paths(tmp_path):
    split = "ref_pdb_ts2"
    write(tmp_path / "vienna/calibration" / split / "vienna_mfe.json",
          {"status": "PASS_NATIVE_PROBABILITY_EVALUATION", "n_sequences": 39, "f1_runs": [0.8] * 3,
           "calibration": {"ece": 0.1, "brier": 0.2, "nll": 0.3},
           "confidence": {"spearman_confidence_vs_structure_f1": 0.4, "aurc_macro": 0.5}})
    write(tmp_path / "calibration_comparisons" / f"{split}.xens2_vienna_mfe.json",
          {"status": "PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD", "matched_n": 29,
           "baseline_clean_n": 39, "xens2_clean_n": 39,
           "calibration": {"xens2_calibrated": {"ece": 0.01, "brier": 0.02, "nll": 0.03},
                           "vienna_mfe_raw": {"ece": 0.1, "brier": 0.2, "nll": 0.3}},
           "structure_confidence": {"xens2_calibrated": {"spearman_confidence_vs_structure_f1": 0.6, "aurc_macro": 0.1},
                                    "vienna_mfe_native": {"spearman_confidence_vs_structure_f1": 0.4, "aurc_macro": 0.5}}})
    cell = build_cell(tmp_path, "vienna_mfe", split)
    assert cell["status"] == "COMPLETE_MATCHED_SENSITIVITY"
    assert cell["own_clean"]["ece"] == 0.1 and cell["matched_xens2"]["ece"] == 0.01
    assert cell["matched_baseline"]["aurc"] == 0.5


def test_rnaformer_uses_shared_raw_key(tmp_path):
    split = "ref_pdb_ts2"
    write(tmp_path / "rnaformer/bprna/calibration" / f"{split}.json",
          {"status": "PASS_NATIVE_PROBABILITY_EVALUATION", "n_sequences": 29, "f1_runs": [0.8] * 3,
           "calibration": {"ece": 0.1, "brier": 0.2, "nll": 0.3},
           "confidence": {"spearman_confidence_vs_structure_f1": 0.4, "aurc_macro": 0.5}})
    write(tmp_path / "calibration_comparisons" / f"{split}.xens2_bprna.json",
          {"status": "PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD", "matched_n": 29,
           "baseline_clean_n": 29, "xens2_clean_n": 29,
           "calibration": {"xens2_calibrated": {"ece": 0.01, "brier": 0.02, "nll": 0.03},
                           "rnaformer_raw": {"ece": 0.1, "brier": 0.2, "nll": 0.3}},
           "structure_confidence": {"xens2_calibrated": {"spearman_confidence_vs_structure_f1": 0.6, "aurc_macro": 0.1},
                                    "rnaformer_native": {"spearman_confidence_vs_structure_f1": 0.4, "aurc_macro": 0.5}}})
    cell = build_cell(tmp_path, "rnaformer_bprna", split)
    assert cell["matched_baseline"]["ece"] == 0.1
    assert cell["matched_baseline"]["aurc"] == 0.5
