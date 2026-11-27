import gzip
import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "tools" / "audit_rfam_release_lineage_20261127.py"
spec = importlib.util.spec_from_file_location("rfam_release_lineage", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_family_first_release_uses_accession_presence_not_created_timestamp():
    assert audit.family_first_release({"14.9": True, "14.10": True, "15.0": True}) == "14.9_OR_EARLIER"
    assert audit.family_first_release({"14.9": False, "14.10": True, "15.0": True}) == "14.10"
    assert audit.family_first_release({"14.9": False, "14.10": False, "15.0": True}) == "15.0"
    assert audit.family_first_release({}) == "UNCHECKED"


def test_seed_alignment_extracts_family_clan_and_exact_region(tmp_path):
    path = tmp_path / "Rfam.seed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("# STOCKHOLM 1.0\n#=GF AC RF00001\n#=GF CL CL00001\n")
        handle.write("AAAP01003520.1/1076-876 GG-CC\n#=GC SS_cons ((..))\n//\n")
    result = audit.seed_alignments(path)
    assert "RF00001" in result["families"]
    assert result["families"]["RF00001"]["clan_acc"] == "CL00001"
    assert result["exact_sequence_regions"]["GGCC"] == {("RF00001", "AAAP01003520.1/1076-876")}


def test_region_key_preserves_rnacentral_taxonomy_suffix():
    assert audit.region_key("URS000075C6AE_59729_1-80.bpseq") == "URS000075C6AE_59729/1-80"
    assert audit.region_key("AAAP01003520.1_1076-876.bpseq") == "AAAP01003520.1/1076-876"
