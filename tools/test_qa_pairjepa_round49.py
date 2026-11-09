import copy

import pytest

from tools.qa_pairjepa_round49 import count_pairs, validate_result


def result_fixture():
    raw = [{"name": "", "seq": "A" * 120, "pairs": [[0, 10], [1, 50]]},
           {"name": "", "seq": "A" * 450, "pairs": [[0, 101]]}]
    predicted = [[[0, 10], [2, 60]], [[0, 101]]]
    per_sequence = [
        {"name": "", "length": 120, "f1": 0.5, "inf": 0.5, "tp": 1, "fp": 1, "fn": 1,
         "span_30": {"tp": 0, "fp": 1, "fn": 1, "f1": 0.0},
         "span_100": {"tp": 0, "fp": 0, "fn": 0, "f1": 1.0}},
        {"name": "", "length": 450, "f1": 1.0, "inf": 1.0, "tp": 1, "fp": 0, "fn": 0,
         "span_30": {"tp": 1, "fp": 0, "fn": 0, "f1": 1.0},
         "span_100": {"tp": 1, "fp": 0, "fn": 0, "f1": 1.0}},
    ]
    predictions = [[{"name": "", "pred_pairs": predicted[0]},
                    {"name": "", "pred_pairs": predicted[1]}] for _ in range(3)]
    result = {"device": "cuda", "gpu_name": "test-gpu",
              "encoder_load": {"strict_identity": True}, "repeats": 3,
              "score_forward_count_per_sequence": 1, "pairjepa_teacher_forward_guard": True,
              "decoder": "nussinov", "prior_weight_override": None,
              "raw_n": 2, "n": 2, "n_long_unsupported": 0, "long_unsupported": [],
              "status": "PASS", "per_sequence": per_sequence, "predictions": predictions,
              "micro": {"tp": 2, "fp": 1, "fn": 1, "f1": 2 / 3,
                        "precision": 2 / 3, "recall": 2 / 3},
              "micro_inf": 2 / 3, "macro_f1": 0.75, "macro_inf": 0.75,
              "calibration": {"n": 1, "ece": 0.0, "brier": 0.0, "nll": 0.0}}
    return result, raw


def test_independent_metrics_and_blank_names():
    result, raw = result_fixture()
    stats = validate_result(result, raw)
    assert stats["f1"] == 2 / 3 and stats["macro_f1"] == 0.75
    assert stats["length_buckets"][1]["n"] == 1
    assert stats["length_buckets"][3]["n"] == 1
    assert stats["span_30"]["macro_f1"] == 0.5
    assert stats["span_100"]["macro_f1"] == 1.0
    assert count_pairs([[0, 101]], [[0, 101]], 100)["tp"] == 1


def test_tampered_counts_and_predictions_fail_closed():
    result, raw = result_fixture()
    changed = copy.deepcopy(result)
    changed["per_sequence"][0]["tp"] = 2
    with pytest.raises(AssertionError):
        validate_result(changed, raw)
    changed = copy.deepcopy(result)
    changed["predictions"][2][0]["pred_pairs"] = []
    with pytest.raises(AssertionError):
        validate_result(changed, raw)
    changed = copy.deepcopy(result)
    changed["predictions"][1].pop()
    with pytest.raises(AssertionError):
        validate_result(changed, raw)


def test_decoder_prior_and_pair_validity_fail_closed():
    result, raw = result_fixture()
    with pytest.raises(AssertionError):
        validate_result(result, raw, decoder="greedy")
    with pytest.raises(AssertionError):
        validate_result(result, raw, prior_override=0.0)
    changed = copy.deepcopy(result)
    changed["predictions"][0][0]["pred_pairs"].append([5, 5])
    with pytest.raises(AssertionError):
        validate_result(changed, raw)
