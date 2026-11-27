#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archiveii_recovery", ROOT / "tools/recover_archiveii_full3966_cpu_baselines_20261127.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_recovery_plan_is_exact_official_minus_clean_without_sequence_rewrite():
    plan = RUNNER.load_plan()
    assert plan["n_records"] == len(plan["rows"]) == 1422
    assert plan["non_acgu_records"] == 0
    assert len({row["name"] for row in plan["rows"]}) == 1422
    assert all(row["sequence_sha256"] for row in plan["rows"])
    assert plan["policy"]["non_acgu"] == "REJECTED_NON_ACGU without normalization or prediction"
    assert plan["policy"]["compute"] == "CPU only; one thread per worker; no GPU visibility"


def test_dot_bracket_and_ct_parsers_preserve_coordinates():
    assert RUNNER.dot_bracket_pairs("((..))", 6) == [(0, 5), (1, 4)]
    sequence = "ACGUAC"
    ct = "\n".join([
        "6 ENERGY = -1.0",
        "1 A 0 2 6 1", "2 C 1 3 5 2", "3 G 2 4 0 3",
        "4 U 3 5 2 4", "5 A 4 6 1 5", "6 C 5 0 0 6",
    ])
    assert RUNNER.parse_ct(ct, sequence) == ([(0, 5), (1, 4)], 1)
    with pytest.raises(ValueError):
        RUNNER.parse_ct(ct.replace("1 A 0 2 6 1", "1 U 0 2 6 1"), sequence)


def test_mxfold2_dbn_requires_exact_sequence_order_and_count():
    rows = [{"name": "a", "seq": "ACGU"}, {"name": "b", "seq": "UGCA"}]
    text = ">1\nACGU\n(()) (-1.0)\n>2\nUGCA\n.... (-0.0)\n"
    assert RUNNER.parse_mxfold2_dbn(text, rows) == {"a": [(0, 3), (1, 2)], "b": []}
    with pytest.raises(ValueError):
        RUNNER.parse_mxfold2_dbn(">1\nUGCA\n....\n>2\nACGU\n(())\n", rows)
    with pytest.raises(ValueError):
        RUNNER.parse_mxfold2_dbn(">1\nACGU\n(())\n", rows)


def test_non_acgu_checkpoint_is_rejection_not_empty_prediction():
    row = {"name": "bad", "seq": "ACGN", "sequence_sha256": "abc"}
    record = RUNNER.rejected_record("mxfold2", row)
    assert record["status"] == "REJECTED_NON_ACGU"
    assert record["pairs"] is None and record["non_acgu_characters"] == ["N"]
    assert RUNNER.validate_saved(record, "mxfold2", row)
    with pytest.raises(ValueError):
        RUNNER.validate_saved({"baseline": "mxfold2", "name": "bad", "sequence_sha256": "abc",
                               "status": "PASS", "pairs": [], "pair_set_sha256": RUNNER.value_sha([])}, "mxfold2", row)


def test_checkpoint_resume_validation_and_partial_consolidation(tmp_path):
    rows = [{"name": "a", "seq": "ACGUAC", "sequence_sha256": "a"},
            {"name": "b", "seq": "ACGUAC", "sequence_sha256": "b"}]
    pairs = [(0, 5)]
    record = RUNNER.success_record("contrafold", rows[0], pairs)
    path = RUNNER.record_path(tmp_path, "contrafold", "a")
    RUNNER.write_json(path, record)
    assert RUNNER.validate_saved(json.loads(path.read_text()), "contrafold", rows[0])
    coverage = RUNNER.consolidate(tmp_path, "contrafold", rows)
    assert coverage["status"] == "PARTIAL"
    assert coverage["n_completed"] == 1 and coverage["remaining_names"] == ["b"]
    assert coverage["n_pass"] == 1 and coverage["coverage"] == 0.5


def test_cpu_environment_and_predictor_metadata_do_not_expose_gpu():
    env = RUNNER.cpu_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert env[key] == "1"
    for baseline in ("contrafold", "mxfold2", "rnastructure_fold", "rnastructure_maxexpect"):
        metadata = RUNNER.metadata(baseline)
        assert metadata["cpu_threads"] == 1 and metadata["gpu_visible"] is False
