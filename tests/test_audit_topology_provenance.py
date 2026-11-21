from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from audit_topology_provenance import audit, topology  # noqa: E402
from verify_topology_provenance import verify  # noqa: E402


def test_single_hairpin_compresses_stack_and_loop():
    result = topology({"seq": "GCGCGCGC", "pairs": [[0, 7], [1, 6], [2, 5]]})
    assert result["n_helices"] == 1
    assert result["n_exterior_roots"] == 1
    assert result["loop_type_counts"] == {"HAIRPIN": 1}
    assert result["signature"] == {
        "exterior_gaps": [0, 0],
        "roots": [{"helix_length": 3, "loop_type": "HAIRPIN", "loop_gaps": [2], "children": []}],
    }


def test_two_exterior_roots_preserve_external_gaps():
    result = topology({"seq": "AAAAAAAAAAAA", "pairs": [[0, 5], [1, 4], [8, 11]]})
    assert result["n_helices"] == 2
    assert result["n_exterior_roots"] == 2
    assert result["signature"]["exterior_gaps"] == [0, 2, 0]
    assert [root["helix_length"] for root in result["signature"]["roots"]] == [2, 1]


def test_multiloop_gaps_and_order_are_exact():
    result = topology({"seq": "A" * 20, "pairs": [[0, 19], [1, 18], [4, 7], [10, 15]]})
    root = result["signature"]["roots"][0]
    assert root["helix_length"] == 2
    assert root["loop_type"] == "MULTILOOP"
    assert root["loop_gaps"] == [2, 2, 2]
    assert result["max_multiloop_order"] == 3
    assert [child["outer"] if "outer" in child else child["helix_length"] for child in root["children"]] == [1, 1]


def test_topology_ignores_sequence_identity_but_not_geometry():
    a = topology({"seq": "AAAAAAAA", "pairs": [[0, 7], [1, 6]]})
    b = topology({"seq": "GCGCGCGC", "pairs": [[0, 7], [1, 6]]})
    c = topology({"seq": "AAAAAAAAA", "pairs": [[0, 7], [1, 6]]})
    assert a["signature_sha256"] == b["signature_sha256"]
    assert a["signature_sha256"] != c["signature_sha256"]


def test_crossing_pairs_are_rejected():
    with pytest.raises(ValueError, match="crossing pair"):
        topology({"seq": "AAAAAAAA", "pairs": [[0, 5], [2, 7]]})


def test_partner_conflicts_are_rejected():
    with pytest.raises(ValueError, match="partner conflict"):
        topology({"seq": "AAAAAAAA", "pairs": [[0, 5], [0, 6]]})


def test_end_to_end_toy_audit_and_independent_replay(tmp_path):
    data_root = tmp_path / "jsonl"
    data_root.mkdir()
    train = [
        {"name": "train_hairpin", "seq": "GCGCGCGC", "pairs": [[0, 7], [1, 6], [2, 5]]},
        {"name": "train_unpaired", "seq": "AAAA", "pairs": []},
    ]
    evaluation = [
        {"name": "same_topology_different_sequence", "seq": "AAAAAAAA", "pairs": [[0, 7], [1, 6], [2, 5]]},
        {"name": "new_unpaired_length", "seq": "AAAAAAAA", "pairs": []},
    ]
    for name, rows in (("bprna_tr1c", train), ("archiveii_embok_clean", evaluation)):
        (data_root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    model_card = tmp_path / "README.md"
    model_card.write_text("datasets: rnacentral, rfam, ensembl-genome-browser, nucleotide\n")
    out = tmp_path / "audit"
    document = audit(data_root, out, model_card)
    summary = document["datasets"]["archiveii_embok_clean"]
    assert summary["status"] == "EXACT"
    assert summary["n_records"] == 2
    assert summary["n_records_with_exact_topology_overlap"] == 1
    assert document["datasets"]["bprna_new"]["status"] == "BLOCKED_DATA"
    assert document["pretraining_exposure"]["status"] == "UNCHECKED"
    result = verify(out / "topology_provenance_audit.json")
    assert result["status"] == "PASS"
    assert result["n_overlap_records"] == 1
    assert all(result["checks"].values())
