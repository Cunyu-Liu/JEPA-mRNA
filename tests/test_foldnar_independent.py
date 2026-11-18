from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rnajepa.fold_structure import structure_metrics
from foldnar_e0 import evaluate_join
from foldnar_independent_check import independent_metrics, verify_task


def _structures(length=6):
    candidates = [(i, j) for i in range(length) for j in range(i + 2, length)]
    for mask in range(1 << len(candidates)):
        pairs = [candidates[k] for k in range(len(candidates)) if mask & (1 << k)]
        used = set()
        valid = True
        for i, j in pairs:
            if i in used or j in used:
                valid = False
                break
            used.update((i, j))
        if not valid:
            continue
        for (i, j), (k, l) in itertools.combinations(pairs, 2):
            if i < k < j < l or k < i < l < j:
                valid = False
                break
        if valid:
            yield pairs


def _assert_same(a, b, path=""):
    if isinstance(b, dict):
        for key in b:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(b, bool):
        assert bool(a) == b, path
    elif isinstance(b, (int, float)):
        assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (path, a, b)


def test_independent_metrics_exhaustive_small_structures():
    structures = list(_structures())
    assert len(structures) == 17
    for truth in structures:
        for pred in structures:
            expected = structure_metrics(pred, truth, length=6)
            actual = independent_metrics(pred, truth, length=6)
            for key in ("pair", "stem", "loop", "loop_types", "multiloop", "junction", "topology",
                        "n_pred_pairs", "n_truth_pairs", "n_pred_stems", "n_truth_stems"):
                _assert_same(expected[key], actual[key], key)


def test_independent_metrics_known_loop_types():
    structures = [
        ([], 16),
        ([(0, 9), (1, 8), (2, 7)], 16),
        ([(0, 9), (3, 6)], 16),
        ([(0, 9), (1, 6)], 16),
        ([(0, 13), (2, 5), (7, 10)], 16),
        ([(0, 15), (1, 6), (8, 13), (9, 12)], 16),
        ([(20, 29), (21, 28), (22, 27)], 40),
    ]
    for pred, length in structures:
        for truth, _ in structures:
            expected = structure_metrics(pred, truth, length=40)
            actual = independent_metrics(pred, truth, length=40)
            for key in ("pair", "stem", "loop", "loop_types", "multiloop", "junction", "topology"):
                _assert_same(expected[key], actual[key], key)


def test_verify_task_toy(tmp_path: Path):
    rows = [
        {"name": "simple", "seq": "ACGUAC", "pairs": [[0, 5], [1, 4]]},
        {"name": "cross", "seq": "ACGUACGU", "pairs": [[0, 5], [2, 7]]},
    ]
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"per_sequence": [
        {"name": "simple", "pred_pairs": [[0, 5], [1, 4]]},
        {"name": "cross", "pred_pairs": []},
    ]}))
    spec = {"prediction_id": "toy/toy", "model": "toy", "split": "toy", "repeat": None, "path": str(pred_path)}
    original = evaluate_join(spec, str(data_path))["rows"]
    report = verify_task(spec, str(data_path), original)
    assert report["n_mismatches"] == 0, report["mismatches"]
    assert report["counts"]["structure"] == 1
    assert report["counts"]["pair_only"] == 1


def test_verify_task_blank_name_positional_alignment(tmp_path: Path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "a", "seq": "ACGUAC", "pairs": [[0, 5]]}) + "\n")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"per_sequence": [
        {"name": "", "length": 6, "n_gt": 1, "f1": 1.0, "pred_pairs": [[0, 5]]},
    ]}))
    spec = {"prediction_id": "blank/test", "model": "blank", "split": "test", "repeat": None, "path": str(pred_path)}
    original = evaluate_join(spec, str(data_path))["rows"]
    report = verify_task(spec, str(data_path), original)
    assert report["n_mismatches"] == 0, report["mismatches"]
    assert report["counts"]["structure"] == 1
