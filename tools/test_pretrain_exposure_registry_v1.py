import gzip

from build_pretrain_exposure_registry_v1 import iter_fasta, normalize, sequence_sha256, stable_id


def test_fasta_normalization_and_source_ids(tmp_path):
    path = tmp_path / "source.fa.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(">source-a full header\nacgt\nNN\n>source-b\nUUUU\n")
    assert list(iter_fasta(path)) == [("source-a full header", "ACGUNN"), ("source-b", "UUUU")]


def test_stable_query_identity_uses_row_and_name():
    row = {"name": "duplicate.name", "seq": "acgt"}
    assert stable_id("bprna_new", 7, row) == "bprna_new:row000007:duplicate.name"
    sequence = normalize(row["seq"])
    assert sequence == "ACGU"
    assert len(sequence_sha256(sequence)) == 64
