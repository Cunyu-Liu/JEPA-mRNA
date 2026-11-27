import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "tools" / "resolve_rnaformer_pdb_metadata_20261127.py"
spec = importlib.util.spec_from_file_location("metadata_resolution", path)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def entity(sequence="ACGU", description="Hatchet ribozyme"):
    return {
        "entity_poly": {"pdbx_seq_one_letter_code_can": sequence,
                        "rcsb_entity_polymer_type": "RNA"},
        "rcsb_polymer_entity": {"pdbx_description": description},
        "rcsb_entity_source_organism": [],
        "rcsb_polymer_entity_container_identifiers": {
            "entry_id": "1ABC", "entity_id": "2", "auth_asym_ids": ["A", "B"], "asym_ids": ["X", "Y"],
        },
    }


def entry(date="2020-01-01T00:00:00Z"):
    return {"rcsb_accession_info": {"initial_release_date": date}}


def test_original_family_text_is_conservative_and_normalized():
    assert resolver.family_matches("Hatchet ribozyme", "Hatchet ribozyme, RNA")
    assert resolver.family_matches("AdoCbl variant RNA", "AdoCbl variant RNA")
    assert not resolver.family_matches("Hatchet ribozyme", "Other RNA")
    assert not resolver.family_matches("unknown", "Hatchet ribozyme")


def test_full_polymer_rna_and_chain_identity_are_required():
    chains, audit = resolver.candidate_chains(entity(), "ACGU", "unknown", entry())
    assert len(chains) == 2
    assert all(chain["match_types"] == ["FULL_POLYMER_EXACT"] for chain in chains)
    chains, _ = resolver.candidate_chains(entity(), "ACG", "unknown", entry())
    assert chains == []


def test_post_rnaformer_entry_is_excluded_without_date_ranking():
    chains, audit = resolver.candidate_chains(entity(), "ACGU", "unknown", entry("2024-01-20T00:00:00Z"))
    assert chains == []
    assert audit["stages"]["pre_or_at_rnaformer_release"] is False


def test_family_filter_does_not_use_pairs():
    chains, audit = resolver.candidate_chains(entity(), "ACGU", "Other family", entry())
    assert chains == []
    assert audit["stages"]["family_text"] is False
