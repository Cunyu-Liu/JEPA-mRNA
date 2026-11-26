import gzip
import importlib.util
from pathlib import Path

def load_module(filename: str):
    path = Path(__file__).parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location("pdb_rfam_lineage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module("audit_pdb_rfam_label_lineage_20261127.py")


def record(**changes):
    row = {
        "name": "A0A0_1-8.bpseq",
        "source": "bpseq:A0A0_1-8.bpseq",
        "seq": "GCAAGCUU",
        "structure": "(......)",
        "pairs": [[0, 7]],
        "n_pairs": 1,
    }
    row.update(changes)
    return row


def test_observed_unobserved_and_unknown_are_distinct():
    assert audit.residue_label_state(False) == "UNOBSERVED_NOT_NEGATIVE"
    assert audit.residue_label_state(None) == "UNCHECKED"
    assert audit.residue_label_state(True).startswith("OBSERVED_UNPAIRED_ONLY")


def test_valid_projection_is_not_a_lineage_upgrade():
    result = audit.audit_record("rfam_fam", 0, record())
    assert result["internal_projection_status"] == "PASS"
    assert result["pair_provenance_status"] == "UNCHECKED"
    assert result["experimental_lineage_status"] == "UNCHECKED"
    assert result["temporal_lineage_status"] == "UNCHECKED"
    assert result["unobserved_residues_are_negative_labels"] == "UNCHECKED"


def test_pair_encoding_conflict_fails_closed():
    result = audit.audit_record("ref_pdb_ts1", 0, record(pairs=[[0, 6]]))
    assert result["internal_projection_status"] == "FAIL"
    assert "STRUCTURE_PAIR_MISMATCH" in result["internal_projection_errors"]


def test_unobserved_dot_positions_do_not_change_pair_hash():
    first = audit.pairs_sha256([[7, 0]])
    second = audit.pairs_sha256([[0, 7]])
    assert first == second


def test_region_key_supports_genomic_and_rnacentral_names():
    assert audit.region_key("AAAP01003520.1_1076-876.bpseq") == "AAAP01003520.1/1076-876"
    assert audit.region_key("URS000075C6AE_59729_1-80.bpseq") == "URS000075C6AE_59729/1-80"
    assert audit.region_key("pdb_ts1#632970") is None


def test_direct_rnacentral_sequence_does_not_invent_family_metadata():
    assert audit.rfam15_status(True, [], ["URS000075C6AE_59729/1-80"]) == "PASS_DIRECT_SEQUENCE_FAMILY_UNCHECKED"
    assert audit.rfam15_status(True, [{"rfam_acc": "RF00001"}], ["x/1-8"]) == "PASS_REGION_SEQUENCE_FAMILY_ONLY"
    assert audit.rfam15_status(False, [], []) == "NOT_EXACT_IN_RFAM15_0"


def test_fasta_match_requires_exact_sequence_or_region(tmp_path):
    path = tmp_path / "Rfam.fa.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(">x/1-4\nACGU\n>y/1-4\nACGA\n>z/1-4\nACG\n")
    hits = audit.fasta_target_map(path, {"ACGU"}, {"y/1-4"})
    assert hits == {"ACGU": ["x/1-4"], "ACGA": ["y/1-4"]}


def test_bad_alphabet_is_not_silently_normalized():
    result = audit.audit_record("rfam_temporal", 0, record(seq="GCAAGCNN", structure="(......)"))
    assert result["internal_projection_status"] == "FAIL"
    assert "BAD_ALPHABET" in result["internal_projection_errors"]
