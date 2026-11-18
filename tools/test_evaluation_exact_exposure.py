import gzip

from audit_evaluation_exact_exposure import audit_sequences
from aggregate_temporal_homology import read_queries


def test_dynamic_evaluation_exact_and_query_count(tmp_path):
    source = tmp_path / "source.fa.gz"
    with gzip.open(source, "wt") as handle:
        handle.write(">a\n" + "A" * 16 + "\n")
    empty = tmp_path / "empty.fa.gz"
    with gzip.open(empty, "wt"):
        pass
    records = [{"sequence": "A" * 16}, {"sequence": "C" * 16}]
    result = audit_sequences(records, {"source": [source], "other": [empty]},
                             {"TR1c": [{"seq": "T" * 16}, {"seq": "A" * 16}], "VL0": []})
    assert result["exact_union_detected"] == 1
    assert result["exact_not_detected"] == 1
    assert result["by_source"] == {"source": 1, "other": 0}
    assert result["downstream_exact_overlap"] == {"TR1c": 1, "VL0": 0}
    fasta = tmp_path / "eval.fa"
    fasta.write_text(">eval_0000|exact_detected=1\n" + "A" * 16 + "\n>eval_0001|exact_detected=0\n" + "C" * 16 + "\n")
    assert len(read_queries(fasta, expected_count=2)) == 2
