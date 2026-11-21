from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pair_error_attribution import (classify_fn, pair_environments, qa_record,
                                    raw_partner_ranks, span_bin)


def evidence(**updates):
    value = {
        "label_issues": [], "mask_issues": [], "model_mask": True,
        "finite_score": True, "finite_posterior": True,
        "raw_mutual_best": False, "posterior_probability": 0.1,
        "training_environment_count": 5,
    }
    value.update(updates)
    return value


def test_label_mask_and_unknown_priority():
    assert classify_fn(evidence(label_issues=["PARTNER_CONFLICT"], raw_mutual_best=True)) == "LABEL"
    assert classify_fn(evidence(model_mask=False, mask_issues=["MIN_LOOP"])) == "MASK"
    assert classify_fn(evidence(finite_score=False)) == "UNKNOWN"


def test_decoder_requires_direct_support():
    assert classify_fn(evidence(raw_mutual_best=True)) == "DECODER"
    assert classify_fn(evidence(posterior_probability=0.5)) == "DECODER"
    assert classify_fn(evidence(posterior_probability=0.499)) == "RAW_SCORE"


def test_data_shift_requires_zero_environment_support():
    assert classify_fn(evidence(training_environment_count=0)) == "DATA_SHIFT"
    assert classify_fn(evidence(training_environment_count=0, raw_mutual_best=True)) == "DECODER"
    assert classify_fn(evidence(training_environment_count=1)) == "RAW_SCORE"


def test_pair_environment_stem_and_loop():
    seq = "GGGAAAACCC"
    environments = pair_environments(seq, [(0, 9), (1, 8), (2, 7)])
    assert environments[(0, 9)]["enclosed_loop_type"] == "STACK"
    assert environments[(0, 9)]["stem_position"] == "OUTER"
    assert environments[(1, 8)]["stem_position"] == "MIDDLE"
    assert environments[(2, 7)]["enclosed_loop_type"] == "HAIRPIN"
    assert environments[(2, 7)]["stem_position"] == "INNER"
    assert environments[(2, 7)]["stem_length"] == 3


def test_raw_partner_ranks_use_strict_competition():
    scores = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], dtype=float)
    mask = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(mask, False)
    assert raw_partner_ranks(scores, mask, 0, 2) == (2, 2)


def test_qa_detects_partner_conflict_and_normalizes_t():
    record = {"name": "x", "seq": "GAAATCCCC", "structure": ".........",
              "pairs": [[0, 7], [0, 8]], "n_pairs": 2, "source": "bpseq:x"}
    qa = qa_record(record)
    assert qa["normalized_sequence"] == "GAAAUCCCC"
    assert "PARTNER_CONFLICT" in qa["pair_label_issues"][(0, 7)]
    assert "PARTNER_CONFLICT" in qa["pair_label_issues"][(0, 8)]


def test_span_bins_are_frozen():
    assert [span_bin(value) for value in (3, 4, 10, 11, 20, 21, 50, 51, 100, 101, 200, 201)] == [
        "LT4", "4-10", "4-10", "11-20", "11-20", "21-50", "21-50",
        "51-100", "51-100", "101-200", "101-200", "201+",
    ]
