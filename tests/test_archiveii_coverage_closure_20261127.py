#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VIENNA = load("vienna_clean", "tools/recover_archiveii_vienna_clean_cpu_20261127.py")
XENS = load("xens_recovery", "tools/recover_archiveii_xens2_gpu_20261127.py")
QUEUE = load("xens_queue", "tools/queue_archiveii_xens2_gpu_recovery_20261127.py")
BASE = load("base_recovery", "tools/recover_archiveii_full3966_cpu_baselines_20261127.py")


def test_vienna_clean_plan_has_exact_official_identities():
    plan = VIENNA.recovery_rows()
    assert len(plan) == 2544
    assert len({row["name"] for row in plan}) == 2544
    assert not any(set(row["seq"]) - set("ACGU") for row in plan)
    assert all(row["sequence_sha256"] for row in plan)


def test_xens2_protocol_blocks_length_and_missing_embedding_without_substitution():
    class Store:
        def get(self, sequence):
            return None if sequence == "ACGU" else [0.0]

    assert XENS.protocol_block_reason({"seq": "A" * 1025}, Store()) == "BLOCKED_PROTOCOL_LENGTH_GT1024"
    assert XENS.protocol_block_reason({"seq": "ACGU"}, Store()) == "BLOCKED_PROTOCOL_EMBEDDING_MISSING"
    assert XENS.protocol_block_reason({"seq": "AAAA"}, Store()) is None
    assert XENS.protocol_block_reason({"seq": "ACGN"}, Store()) == "REJECTED_NON_ACGU"
    plan = json.loads(XENS.PLAN_PATH.read_text())["rows"]
    assert sum(len(row["seq"]) > 1024 for row in plan) == 16


def test_gpu_admission_thresholds_are_conservative_and_persistent():
    static = QUEUE.static_checkpoint_mib()
    assert static > 4000
    assert QUEUE.probe_requirement_mib(static) == static + 2048
    assert QUEUE.production_requirement_mib(1000) == 3548
    assert QUEUE.production_requirement_mib(10000) == 17048


def test_xens2_checkpoint_validation_and_partial_coverage(tmp_path):
    rows = [{"name": "a", "seq": "ACGU", "sequence_sha256": "a"},
            {"name": "b", "seq": "ACGU", "sequence_sha256": "b"}]
    good = XENS.success_record(rows[0], [], {"device": "cuda", "device_name": "test"})
    BASE.write_json(BASE.record_path(tmp_path, "xens2", "a"), good)
    blocked = XENS.blocked_record(rows[1], "BLOCKED_PROTOCOL_EMBEDDING_MISSING")
    BASE.write_json(BASE.record_path(tmp_path, "xens2", "b"), blocked)
    coverage = XENS.consolidate(tmp_path, rows)
    assert coverage["status"] == "COMPLETE"
    assert coverage["n_pass"] == 1 and coverage["n_blocked"] == 1
    assert coverage["blocked_reason_counts"] == {"BLOCKED_PROTOCOL_EMBEDDING_MISSING": 1}
    with pytest.raises(ValueError):
        XENS.validate_saved({"name": "a", "sequence_sha256": "wrong", "status": "PASS", "pairs": []}, rows[0])
