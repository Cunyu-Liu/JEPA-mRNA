import hashlib
import json

import pytest

from aggregate_temporal_homology import SOURCES, aggregate


def sequence(index):
    return "AAAAAAAAA" + "".join("ACGU"[(index >> (2 * offset)) & 3] for offset in range(7))


def fixture(tmp_path, coverage=16, target_length=16):
    fasta = tmp_path / "queries.fasta"
    fasta.write_text("".join(f">q{index}|exact_detected=0\n{sequence(index)}\n" for index in range(1131)))
    for source in SOURCES:
        directory = tmp_path / source
        directory.mkdir()
        result = directory / "matches.tsv"
        result.write_text(f"q0|exact_detected=0\ttarget\t80\t{coverage}\t16\t{target_length}\t1\t16\t1\t{coverage}\t0\t30\n" if source == SOURCES[0] else "")
        manifest = {"status": "PASS_SEARCH_COMPLETED_AGGREGATION_PENDING", "source": source,
                    "mmseqs_version": "17.b804f", "target_sha256": "target",
                    "command": ["mmseqs", "easy-search", "--min-seq-id", "0.8", "--cov-mode", "0",
                                "-c", "0.8", "--max-seqs", "300"],
                    "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}
        (directory / "search_manifest.json").write_text(json.dumps(manifest))
    return fasta


def test_union_conserves_unknown_queries(tmp_path):
    result = aggregate(fixture(tmp_path), tmp_path)
    assert result["n_queries"] == result["detected_by_any_source"] + result["not_detected_by_any_source"]
    assert result["detected_by_any_source"] == 1
    assert result["not_detected_by_any_source"] == 1130
    unknown = result["query_records"][1]
    assert unknown["interpretation"] == "UNKNOWN_NOT_CLEAN"
    assert unknown["exact_detected"] is False


def test_bidirectional_coverage_is_enforced(tmp_path):
    with pytest.raises(AssertionError):
        aggregate(fixture(tmp_path, coverage=12), tmp_path)


def test_exact_match_missing_from_homology_is_blocked(tmp_path):
    fasta = fixture(tmp_path)
    text = fasta.read_text().replace(">q1|exact_detected=0", ">q1|exact_detected=1")
    fasta.write_text(text)
    result = aggregate(fasta, tmp_path)
    assert result["status"] == "BLOCKED_EXACT_NOT_HOMOLOGY_DETECTED"
    assert result["exact_detected_not_homology_detected"] == ["q1|exact_detected=1"]


def test_target_coverage_postfilter_preserves_raw_count(tmp_path):
    result = aggregate(fixture(tmp_path, target_length=32), tmp_path)
    assert result["detected_by_any_source"] == 0
    source = result["sources"][SOURCES[0]]
    assert source["reported_matches"] == 1
    assert source["excluded_target_coverage"] == 1
    assert source["excluded_examples"][0]["tlen"] == "32"
