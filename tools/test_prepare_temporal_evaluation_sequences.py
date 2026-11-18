import json

import prepare_temporal_evaluation_sequences as preparation


def test_sequence_identity_and_metadata_mismatch(tmp_path):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "collection.json").write_text("{}")
    metadata = [{"entry_id": "P1", "entity_id": "1", "identifier": "P1_1", "sequence": "A" * 16},
                {"entry_id": "P1", "entity_id": "2", "identifier": "P1_2", "sequence": "C" * 16},
                {"entry_id": "P2", "entity_id": "1", "identifier": "P2_1", "sequence": "N" * 16}]
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"records": metadata}))
    documents = {"P1": [], "P2": []}
    for pdb, entity, chain, sequence in [("P1", "1", "A", "A" * 16), ("P1", "2", "B", "A" * 16), ("P2", "1", "C", "U" * 16)]:
        documents[pdb].append({"id": f"{pdb}_{chain}", "pdb_id": pdb, "entity_ids": [entity], "sequence": sequence})
    for pdb, chains in documents.items():
        directory = root / pdb
        directory.mkdir()
        (directory / "labels.json").write_text(json.dumps({"chains": chains}))
    preparation.EXPECTED_METADATA = preparation.sha256(metadata_path)
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps({"status": "PASS_TEMPORAL_LABEL_COLLECTION_QA_HOMOLOGY_PENDING",
                              "metadata_sha256": preparation.EXPECTED_METADATA,
                              "collection_sha256": preparation.sha256(root / "collection.json"),
                              "n_accepted_chains": 3, "n_unique_accepted_sequences": 2,
                              "metadata_sequence_differences": [{"id": "P2_C"}]}))
    records = preparation.prepare(root, metadata_path, qa)
    assert len(records) == 2
    assert sum(len(row["chain_ids"]) for row in records) == 3
    restored = next(row for row in records if row["sequence"] == "U" * 16)
    assert restored["metadata_records"][0]["metadata_sequence_mismatch"] is True
    duplicate = next(row for row in records if row["sequence"] == "A" * 16)
    assert duplicate["chain_ids"] == ["P1_A", "P1_B"]
