from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from foldnar_e0 import decide_gate, evaluate_join, flatten_row


def test_evaluate_join_and_same_set_gap(tmp_path: Path) -> None:
    data = [
        {"name": "simple", "seq": "ACGUAC", "pairs": [[0, 5], [1, 4]]},
        {"name": "cross", "seq": "ACGUACGU", "pairs": [[0, 5], [2, 7]]},
        {"name": "not_predicted", "seq": "ACGU", "pairs": []},
    ]
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text("".join(json.dumps(x) + "\n" for x in data))
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"n_sequences": 2, "per_sequence": [
        {"name": "simple", "pred_pairs": [[0, 5], [1, 4]]},
        {"name": "cross", "pred_pairs": []},
    ]}))
    spec = {"prediction_id": "toy/toy", "model": "toy", "split": "toy", "repeat": None, "path": str(pred_path)}
    result = evaluate_join(spec, str(data_path))
    assert result["summary"]["n_joined_names"] == 2
    assert result["summary"]["n_structure_valid"] == 1
    assert result["summary"]["coverage_of_truth"] == pytest.approx(2 / 3)
    assert result["summary"]["macro"]["pair_f1"] == pytest.approx(0.5)
    assert result["summary"]["macro"]["topology_f1"] == 1
    # The gap uses only the one sequence where both pair and topology are defined.
    assert result["summary"]["macro"]["pair_f1_minus_topology_f1"] == 0
    cross = next(r for r in result["rows"] if r["name"] == "cross")
    assert cross["status"] == "STRUCTURE_UNDEFINED"
    assert cross["pair"]["fn"] == 2
    flat = flatten_row(result["rows"][0])
    assert "pair_f1" in flat and "topology_ged" in flat


def _run(model: str, split: str, topology: float = 0.7) -> dict:
    return {"model": model, "split": split, "repeat": None, "coverage_of_truth": 1,
            "n_structure_valid": 10,
            "macro": {"topology_f1": topology, "stem_f1": 0.7, "pair_f1_minus_topology_f1": 0.1,
                      "multiloop_f1": 0.6, "junction_f1": 0.6}}


def test_gate_requires_evidence_and_two_split_trigger(tmp_path: Path) -> None:
    runs = []
    for model in ("xens2", "mxfold2"):
        runs += [_run(model, "bprna_new"), _run(model, "bprna_ts0")]
    gate = decide_gate(runs, tmp_path / "missing_leakage.json")
    assert gate["evidence"]["sufficient"] is True
    assert gate["trigger"]["distinct_rule1_splits"] == 2
    assert gate["e0_decision"] == "GO_E1"
    assert gate["execution_decision"] == "PENDING_PHASE1_LEAKAGE_AUDIT"


def test_gate_blocks_when_second_system_missing(tmp_path: Path) -> None:
    gate = decide_gate([_run("xens2", "bprna_new"), _run("xens2", "bprna_ts0")], tmp_path / "none.json")
    assert gate["evidence"]["sufficient"] is False
    assert gate["e0_decision"] == "BLOCKED_INSUFFICIENT_EVIDENCE"
    assert gate["execution_decision"] == "DO_NOT_TRAIN"


def test_gate_no_trigger_does_not_train(tmp_path: Path) -> None:
    runs = []
    for model in ("xens2", "mxfold2"):
        for split in ("bprna_new", "bprna_ts0"):
            run = _run(model, split, topology=1)
            run["macro"].update({"stem_f1": 1, "pair_f1_minus_topology_f1": 0,
                                 "multiloop_f1": 1, "junction_f1": 1})
            runs.append(run)
    gate = decide_gate(runs, tmp_path / "none.json")
    assert gate["e0_decision"] == "NO_GO"
    assert gate["execution_decision"] == "DO_NOT_TRAIN"


def test_blank_names_require_verified_positional_alignment(tmp_path: Path) -> None:
    data = [{"id": "a", "seq": "ACGUAC", "pairs": [[0, 5]]},
            {"id": "b", "seq": "ACGU", "pairs": []}]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("".join(json.dumps(x) + "\n" for x in data))
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"per_sequence": [
        {"name": "", "length": 6, "n_gt": 1, "f1": 1.0, "pred_pairs": [[0, 5]]},
        {"name": "", "length": 4, "n_gt": 0, "f1": 1.0, "pred_pairs": []},
    ]}))
    spec = {"prediction_id": "blank/test", "model": "blank", "split": "test", "repeat": None, "path": str(pred_path)}
    result = evaluate_join(spec, str(data_path))
    assert result["summary"]["n_joined_names"] == 2
    assert result["summary"]["coverage_of_truth"] == 1
    assert result["summary"]["join"]["method"] == "positional_verified_by_length_n_gt_and_f1"
    assert [r["name"] for r in result["rows"]] == ["a", "b"]


def test_blank_names_reject_unverified_alignment(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "a", "seq": "ACGUAC", "pairs": [[0, 5]]}) + "\n")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"per_sequence": [
        {"name": "", "length": 6, "n_gt": 1, "f1": 0.0, "pred_pairs": [[0, 5]]},
    ]}))
    spec = {"prediction_id": "blank/test", "model": "blank", "split": "test", "repeat": None, "path": str(pred_path)}
    with pytest.raises(ValueError, match="F1 mismatch"):
        evaluate_join(spec, str(data_path))
