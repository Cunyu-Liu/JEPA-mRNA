from __future__ import annotations

import hashlib
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from foldnar_phase1 import (
    exact_overlap, existing_exposure_audit, load_per_sequence_homology, matched_length_subset, near_duplicate_screen,
    normalize_sequence, pair_problems, read_dataset,
)


def test_pair_problems_and_crossing() -> None:
    checked = pair_problems([[0, 5], [2, 7], [8, 8], [0, 5], [-1, 3]], 10)
    assert checked["crossing"] is True
    assert checked["problems"]["crossing_pair"] == 2
    assert checked["problems"]["self_pair"] == 1
    assert checked["problems"]["duplicate_pair"] == 1
    assert checked["problems"]["out_of_range"] == 1


def test_pair_problems_multi_paired_nucleotide() -> None:
    checked = pair_problems([[0, 5], [0, 7]], 10)
    assert checked["problems"]["multi_paired_nucleotide"] == 1
    assert checked["problems"]["invalid_record"] == 1


def _record(dataset_id: str, name: str, seq: str, role: str) -> tuple[str, dict]:
    seq = normalize_sequence(seq)
    return dataset_id, {"spec": {"role": role}, "records": [
        {"id": name, "length": len(seq), "seq": seq, "sha256": hashlib.sha256(seq.encode()).hexdigest(), "family": None, "topology": None}
    ]}


def test_exact_overlap_detects_u_t_equivalence() -> None:
    train = _record("train", "a", "ACGT", "train")
    test = _record("test", "b", "ACGU", "test")
    result = exact_overlap(dict([train, test]))
    assert result["n_shared_sequences"] == 1
    assert result["matrix"][0]["status"] == "EXACT_OVERLAP"


def test_validation_split_is_in_leakage_audits() -> None:
    train = _record("train", "a", "ACGU", "train")
    val = _record("dev", "b", "ACGT", "val")
    test = _record("test", "c", "AAAA", "test")
    datasets = dict([train, val, test])
    exact = exact_overlap(datasets)
    assert {row["test"] for row in exact["matrix"]} == {"dev", "test"}
    assert next(row for row in exact["matrix"] if row["test"] == "dev")["n_overlap_sequences"] == 1
    near = near_duplicate_screen(datasets)
    assert near["n_test_sequences"] == 2


def test_near_duplicate_screen_hand_example() -> None:
    seq = ("ACGU" * 30)[:110]
    mutated = seq[:53] + ("A" if seq[53] != "A" else "C") + seq[54:]
    train = _record("train", "train-a", seq, "train")
    test = _record("test", "test-b", mutated, "test")
    result = near_duplicate_screen(dict([train, test]))
    assert result["n_hits"] >= 1
    assert result["examples"][0]["sequence_identity"] >= 0.9


def test_read_dataset_qa_and_numeric_output(tmp_path: Path) -> None:
    rows = [
        {"name": "simple", "seq": "ACGUAC", "pairs": [[0, 5], [1, 4]], "family": "F1"},
        {"name": "cross", "seq": "ACGUACGU", "pairs": [[0, 5], [2, 7]], "family": "F2"},
    ]
    path = tmp_path / "toy.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec = {"dataset_id": "toy", "role": "test", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report, records, numeric = read_dataset(spec)
    assert report["n_records"] == 2
    assert report["quality_counts"]["crossing_pair"] == 2
    assert report["topology"]["n_valid_noncrossing"] == 1
    assert report["family"]["n_unique"] == 2
    assert numeric[0][:4] == (6, 2, 0, 1)
    assert numeric[0][4:] == (3, 2, 8, 2, 5, 1, 2, 2, 0, 0, 0)
    assert report["pair_conservation"]["unaccounted_pair_deletion"] == 0
    assert report["gc_fraction"]["mean"] == 0.5
    assert report["pair_span"]["weighted_mean_span"] == 4.5
    assert report["helices"]["weighted_mean_length"] == 2
    assert numeric[1][3] == -1
    assert len(records) == len(numeric) == 2


def test_near_duplicate_short_sequence_not_overcalled() -> None:
    train = _record("train", "short", "ACGU", "train")
    test = _record("test", "different", "ACGA", "test")
    result = near_duplicate_screen(dict([train, test]))
    assert result["n_hits"] == 0


def test_short_span_is_recorded_not_silently_deleted() -> None:
    checked = pair_problems([[0, 3]], 6)
    assert checked["problems"]["span_lt_4"] == 1
    assert checked["pairs"] == [(0, 3)]


def test_length_matching_is_deterministic_and_balanced() -> None:
    train = []
    for name, length in (("b", 60), ("a", 60), ("c", 65), ("d", 130), ("e", 10)):
        seq = "A" * length
        train.append({"id": name, "length": length, "sha256": hashlib.sha256(seq.encode()).hexdigest()})
    test = []
    for name, length in (("x", 60), ("y", 65), ("z", 130)):
        seq = "C" * length
        test.append({"id": name, "length": length, "sha256": hashlib.sha256(seq.encode()).hexdigest()})
    result = matched_length_subset(train, test)
    assert result["n_matched_per_group"] == 3
    zero = next(row for row in result["bins"] if row["length_bin"] == "0-63")
    assert zero["n_per_group"] == 1
    assert zero["train"][0]["id"] == "a"


def test_existing_exposure_evidence_is_hash_checked_and_not_overcleared(tmp_path: Path) -> None:
    reporting = tmp_path / "artifacts/v2/reporting"
    reporting.mkdir(parents=True)
    audit_dir = tmp_path / "manifests/v2/data_audit"
    audit_dir.mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = {
        "status": "PASS_KNOWN_SOURCE_HOMOLOGY_UNION_NT_AND_FAMILY_LIMITS_REMAIN",
        "scientific_scope": "known-source 80% identity audit", "n_queries": 2, "detected_by_any_source": 2, "not_detected_by_any_source": 0,
        "sources": {"Rfam": {"summary_path": str(source), "summary_sha256": source_hash}},
        "queries": {"bprna_new": {"n_queries": 2, "detected_by_any_source": 2, "not_detected_by_any_source": 0}},
        "limitations": ["non-detection is unknown"],
    }
    summary_path = reporting / "homology_union_summary.json"
    summary_path.write_text(json.dumps(summary) + "\n")
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (reporting / "homology_union_independent_qa.json").write_text(json.dumps({
        "status": "PASS_HOMOLOGY_UNION_INDEPENDENT_QA", "summary_sha256": summary_hash}) + "\n")
    tr1c_hash = "abc"
    data_audit = {
        "status": "PASS_AUDIT_WITH_EXPLICIT_LIMITS",
        "tr1c": {"sha256": tr1c_hash, "attrition_totals": {"n_dropped_crossing": 1}},
        "evaluation_exact_overlap": {"bprna_new.jsonl": {
            "n_unique_exact_overlap": 0, "n_80pct_cluster_overlap": 1}},
        "limits": ["cluster overlap is not exact overlap"],
    }
    (audit_dir / "data_exposure_audit.json").write_text(json.dumps(data_audit) + "\n")
    registry = [
        {"dataset_id": "bprna_tr1c", "role": "train", "status": "FROZEN", "sha256": tr1c_hash},
        {"dataset_id": "bprna_new", "role": "test", "status": "FROZEN", "sha256": "def"},
        {"dataset_id": "chanrg_test", "role": "test", "status": "FROZEN", "sha256": "ghi"},
    ]
    result = existing_exposure_audit(tmp_path, registry)
    assert result["status"] == "PASS_EXISTING_EVIDENCE_INTEGRATED_UNKNOWNS_REMAIN"
    rows = {row["dataset_id"]: row for row in result["pretraining"]["datasets"]}
    assert rows["bprna_new"]["status"] == "HOMOLOGY"
    assert rows["chanrg_test"]["status"] == "UNCHECKED"
    assert result["downstream_cluster_evidence"]["evaluation"][0]["cluster_status"] == "HOMOLOGY"
    assert result["upstream_cleaning_evidence"]["tr1c_attrition_totals"]["n_dropped_crossing"] == 1


def test_per_sequence_homology_join_uses_hashes_and_preserves_unknown(tmp_path: Path) -> None:
    union = tmp_path / "query_homology_union.jsonl.gz"
    union_rows = [
        {"query_id": "Q0", "detected_sources": ["rfam"], "interpretation": "DETECTED"},
        {"query_id": "Q1", "detected_sources": [], "interpretation": "NOT_DETECTED_LIMITED"},
    ]
    with gzip.open(union, "wt") as handle:
        for row in union_rows:
            handle.write(json.dumps(row) + "\n")
    index = tmp_path / "query_index.jsonl.gz"
    index_rows = [
        {"query_id": "Q0", "sequence_sha256": "aaa", "source_sets": ["bprna_new"]},
        {"query_id": "Q1", "sequence_sha256": "bbb", "source_sets": ["bprna_new", "bprna_ts0"]},
    ]
    with gzip.open(index, "wt") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    manifest = tmp_path / "query_manifest.json"
    manifest.write_text(json.dumps({"query_index": str(index), "query_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "union_per_query": str(union), "union_per_query_sha256": hashlib.sha256(union.read_bytes()).hexdigest(),
        "query_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    mapping, metadata = load_per_sequence_homology(summary)
    assert mapping[("bprna_new", "aaa")]["status"] == "HOMOLOGY"
    assert mapping[("bprna_new", "bbb")]["status"] == "NOT_DETECTED_UNKNOWN"
    assert mapping[("bprna_ts0", "bbb")]["status"] == "NOT_DETECTED_UNKNOWN"
    assert metadata["status"] == "PASS_PER_SEQUENCE_HOMOLOGY_IDENTITY_JOIN"
    assert metadata["n_dataset_sequence_links"] == 3
