#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data/ts0_official_coverage_recovery_20261127"
MANIFEST = json.loads((SOURCE / "recovery_manifest.json").read_text())
PANEL = ROOT / "source_data/ts0_testsetb_official_reference_20261127/bprna_ts0_official_1305.jsonl.gz"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_document(relative):
    with gzip.open(ROOT / relative, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_full_and_partial_recovery_identities():
    assert MANIFEST["status"] == "PASS_7_FULL1305_IDENTITIES_2_PARTIAL"
    assert len(MANIFEST["new_full_coverage_identities"]) == 7
    assert MANIFEST["recovered_partial_identities"] == ["rnastructure_fold/bprna_ts0", "rnastructure_maxexpect/bprna_ts0"]
    assert len(MANIFEST["prediction_entries"]) == 9
    assert not MANIFEST["external_test_model_selection"]
    assert MANIFEST["supplement"]["projected_records_rerun"] == 0
    assert MANIFEST["supplement"]["device"] == "CPU_ONLY" and not MANIFEST["supplement"]["gpu_used"]


def test_per_id_official_release_and_original_coverage_audit():
    rows = MANIFEST["per_id_audit"]
    assert len(rows) == 17
    assert [row["name"] for row in rows if row["truth_empty"]] == ["bpRNA_SRP_192.bpseq"]
    blocked = {"bpRNA_RFAM_10644.bpseq", "bpRNA_RFAM_18822.bpseq", "bpRNA_RFAM_18935.bpseq"}
    for row in rows:
        assert row["existing_frozen_coverage"] == []
        assert row["vienna_raw_strict_coverage"] == []
        assert row["bpfold_official_release"]["covered"] is True
        expected = {"rnastructure_fold/bprna_ts0", "rnastructure_maxexpect/bprna_ts0"} if row["name"] in blocked else set()
        assert set(row["blocked_recovered_identities"]) == expected
        assert len(row["final_full_coverage_identities"]) == (7 if expected else 9)


def test_normalized_prediction_identity_and_sequence_sha():
    with gzip.open(PANEL, "rt", encoding="utf-8") as handle:
        panel = {row["name"]: row for row in map(json.loads, handle)}
    for entry in MANIFEST["prediction_entries"]:
        path = ROOT / entry["path"]
        assert sha256(path) == entry["sha256"]
        document = load_document(entry["path"])
        assert document["prediction_id"] == entry["prediction_id"]
        assert document["data_sha256"] == MANIFEST["official_panel"]["sha256"]
        assert document["n_sequences"] == entry["n_prediction_records"]
        rows = document["per_sequence"]
        assert len(rows) == entry["n_prediction_records"]
        assert len({row["name"] for row in rows}) == len(rows)
        for row in rows:
            ref = panel[row["name"]]
            assert row["seq"] == ref["seq"]
            assert row["sequence_sha256"] == ref["sequence_sha256"] == hashlib.sha256(row["seq"].encode()).hexdigest()


def test_cpu_supplement_repeat_and_blocker_contract():
    supplement = json.loads((SOURCE / "missing17_cpu_predictions.json").read_text())
    assert supplement["status"] == "PASS_17_AUDITED_CPU_WITH_EXPLICIT_UNSUPPORTED"
    assert supplement["missing_records"] == 17 and supplement["deterministic_repeats"] == 3
    assert supplement["cuda_visible_devices"] == "" and not supplement["gpu_used"]
    for system, result in supplement["systems"].items():
        assert len(result["repeat_prediction_sha256"]) == 3
        assert len(set(result["repeat_prediction_sha256"])) == 1
        assert len(result["frozen_replay"] if "frozen_replay" in result else supplement["frozen_replay"][system]) == 6
        assert all(row["pairs_exact"] for row in supplement["frozen_replay"][system])
        blocked = result.get("blocked_records", [])
        if system.startswith("rnastructure_"):
            assert len(blocked) == 3
            assert all(row["status"] == "BLOCKED_OFFICIAL_TOOL_INVALID_NUCLEOBASE" for row in blocked)
        else:
            assert blocked == []
            assert len(result["per_sequence"]) == 17


def test_source_asset_identities_are_frozen():
    assert sha256(MANIFEST["release"]["path"]) == MANIFEST["release"]["sha256"]
    for item in MANIFEST["raw_sources"].values():
        assert sha256(item["path"]) == item["sha256"]
    assert sha256(SOURCE / "missing17_cpu_predictions.json") == MANIFEST["supplement"]["sha256"]
