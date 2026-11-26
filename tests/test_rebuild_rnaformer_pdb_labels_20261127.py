import importlib.util
import json
from pathlib import Path


def load_module(filename: str):
    path = Path(__file__).parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location("pdb_rcsb_rebuild", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebuild = load_module("rebuild_rnaformer_pdb_labels_20261127.py")


def test_search_threshold_is_candidate_recall_not_exact_match():
    payload = rebuild.search_payload("ACGU")
    assert payload["query"]["parameters"]["identity_cutoff"] == 0.8
    assert payload["return_type"] == "polymer_entity"
    assert payload["request_options"]["return_all_hits"] is True


def test_subsequence_is_only_a_mmcif_candidate_filter():
    assert rebuild.is_subsequence("ACG", "AACGU")
    assert not rebuild.is_subsequence("ACG", "AGU")
    assert not rebuild.is_subsequence("ACGU", "ACG")


def test_empty_search_is_frozen_as_unchecked_candidate_set(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, "post", lambda *args: (204, b""))
    result = rebuild.freeze_one_search("ACGU", tmp_path)
    assert result["http_status"] == 204
    assert result["candidate_entity_ids"] == []
    assert result["response_sha256"] == rebuild.sha256_bytes(b"")
    assert (tmp_path / f"{rebuild.sha256_text('ACGU')}.json").exists()


def test_atom_site_hetatm_polymer_residue_is_observed(tmp_path):
    path = tmp_path / "1ABC.cif"
    path.write_text("""data_1ABC
loop_
_entity_poly.entity_id
_entity_poly.pdbx_seq_one_letter_code_can
1
;ACG
;
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_ins_code
D 1 1 A 1 D .
D 1 2 C ? D .
D 1 3 G 3 D .
loop_
_atom_site.group_PDB
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
HETATM 1 1 D
ATOM 1 3 D
""")
    chain = rebuild.mmcif_chain_sequences(path, {"1"})[0]
    assert chain["full_sequence"] == "ACG"
    assert chain["observed_mask"] == [True, False, True]
    assert chain["observed_sequence"] == "AG"
    assert chain["scheme_atom_observation_disagreements"] == []


def test_dssr_pairs_remap_observed_and_full_coordinates():
    chain = {
        "auth_chain": "D",
        "scheme_rows": [
            {"auth_seq": "1", "insertion": "?", "seq_id": 1},
            {"auth_seq": "2", "insertion": "?", "seq_id": 2},
            {"auth_seq": "3", "insertion": "?", "seq_id": 3},
        ],
        "observed_mask": [True, False, True],
    }
    result = rebuild.dssr_chain_pairs({"pairs": [{"nt1": "D.A1", "nt2": "D.G3"}]}, chain)
    assert result["status"] == "PASS"
    assert result["full_pairs"] == [[0, 2]]
    assert result["observed_pairs"] == [[0, 1]]


def test_unobserved_dssr_endpoint_fails_instead_of_becoming_negative():
    chain = {
        "auth_chain": "D",
        "scheme_rows": [{"auth_seq": "1", "insertion": "?", "seq_id": 1},
                        {"auth_seq": "2", "insertion": "?", "seq_id": 2}],
        "observed_mask": [False, True],
    }
    result = rebuild.dssr_chain_pairs({"pairs": [{"nt1": "D.A1", "nt2": "D.G2"}]}, chain)
    assert result["status"] == "DSSR_MAPPING_ERROR_UNCHECKED"
    assert result["full_pairs"] == []


def test_search_response_requires_conserved_unique_ids(tmp_path, monkeypatch):
    raw = json.dumps({"total_count": 2, "result_set": [{"identifier": "1ABC_1"}]}).encode()
    monkeypatch.setattr(rebuild, "post", lambda *args: (200, raw))
    try:
        rebuild.freeze_one_search("ACGU", tmp_path)
    except AssertionError:
        pass
    else:
        raise AssertionError("inconsistent search response was accepted")
