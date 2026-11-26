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


def test_search_response_requires_conserved_unique_ids(tmp_path, monkeypatch):
    raw = json.dumps({"total_count": 2, "result_set": [{"identifier": "1ABC_1"}]}).encode()
    monkeypatch.setattr(rebuild, "post", lambda *args: (200, raw))
    try:
        rebuild.freeze_one_search("ACGU", tmp_path)
    except AssertionError:
        pass
    else:
        raise AssertionError("inconsistent search response was accepted")
