from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_homology_mmseqs import parse_m8, summarize_hits


def maps():
    query_map = {"q": {"dataset": "eval", "original_id": "query", "sequence_sha256": "qh"},
                   "q2": {"dataset": "eval", "original_id": "query2", "sequence_sha256": "qh2"}}
    target_map = {"t": {"dataset": None, "original_id": "target"}}
    return query_map, target_map


def test_parse_m8_requires_identity_and_bidirectional_coverage(tmp_path):
    query_map, target_map = maps()
    lengths = {"q": 100, "q2": 100}
    target_lengths = {"t": 100}
    path = tmp_path / "hits.tsv"
    # identity pass, query coverage 0.79 -> reject
    path.write_text("q\tt\t0.95\t100\t0\t0\t0\t78\t0\t99\t1e-20\t200\n")
    assert parse_m8(path, lengths, target_lengths, query_map, target_map) == []
    # both coverages 0.80 and identity 0.80 -> accept
    path.write_text("q\tt\t0.80\t80\t0\t0\t0\t79\t0\t79\t1e-20\t200\n")
    hits = parse_m8(path, lengths, target_lengths, query_map, target_map)
    assert len(hits) == 1 and hits[0]["query_coverage"] == 0.8


def test_summary_is_unknown_not_clean_when_no_hit():
    query_map, _ = maps()
    result = summarize_hits([], query_map)["eval"]
    assert result["contract_status"] == "NOT_DETECTED_UNKNOWN"
    assert result["n_query_records"] == 2


def test_summary_deduplicates_query_records_and_sequences(tmp_path):
    query_map, target_map = maps()
    lengths = {"q": 100, "q2": 100}
    target_lengths = {"t": 100}
    path = tmp_path / "hits.tsv"
    path.write_text("q\tt\t0.90\t100\t0\t0\t0\t99\t0\t99\t1e-20\t200\n" * 2)
    hits = parse_m8(path, lengths, target_lengths, query_map, target_map)
    result = summarize_hits(hits, query_map)["eval"]
    assert result["contract_status"] == "HOMOLOGY"
    assert result["n_hit_records"] == 1
    assert result["n_hit_unique_sequences"] == 1
    assert result["n_alignments_passing_postfilter"] == 2
