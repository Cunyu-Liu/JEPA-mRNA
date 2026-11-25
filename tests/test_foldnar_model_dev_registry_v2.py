import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_foldnar_model_dev_registry_v2 import (
    CHANRG_MATCHED_PANEL, CHANRG_RAW_PANEL, CLUSTER_PANEL, FAMILY_PANEL,
    cluster_run_route, matched_chanrg_ids, stable_id, validate_legal_inputs,
)


class FoldNarModelDevRegistryV2Test(unittest.TestCase):
    def test_stable_ids_are_unique_for_duplicate_names(self):
        self.assertEqual(stable_id("chanrg_train", 3, "A_1-20"), "chanrg_train:row000003:A_1-20")
        self.assertNotEqual(stable_id("chanrg_train", 3, "same"), stable_id("chanrg_train", 4, "same"))

    def test_matched_clean_excludes_both_sides_of_every_shared_cluster(self):
        records = [
            {"dataset": "chanrg_train", "source_row_index": 0, "id": "t0", "cluster80": "shared"},
            {"dataset": "chanrg_train", "source_row_index": 1, "id": "t1", "cluster80": "train_only"},
            {"dataset": "chanrg_validation", "source_row_index": 0, "id": "v0", "cluster80": "shared"},
            {"dataset": "chanrg_validation", "source_row_index": 1, "id": "v1", "cluster80": "validation_only"},
            {"dataset": "tr1c", "source_row_index": 0, "id": "ignored", "cluster80": "shared"},
        ]
        matched, shared = matched_chanrg_ids(records)
        self.assertEqual(shared, {"shared"})
        self.assertEqual(matched["chanrg_train"], ["chanrg_train:row000001:t1"])
        self.assertEqual(matched["chanrg_validation"], ["chanrg_validation:row000001:v1"])

    def test_cluster_trained_run_has_only_cluster_dev_route(self):
        self.assertEqual(cluster_run_route(CLUSTER_PANEL)[0], "ELIGIBLE")
        for panel in (FAMILY_PANEL, CHANRG_RAW_PANEL, CHANRG_MATCHED_PANEL):
            self.assertEqual(cluster_run_route(panel)[0], "INELIGIBLE")
        self.assertIn("never", cluster_run_route(CHANRG_RAW_PANEL)[1])
        self.assertIn("UNCHECKED", cluster_run_route(CHANRG_MATCHED_PANEL)[1])

    def test_legal_scope_rejects_external_test_and_unexpected_assets(self):
        summary = {"input_identities": {
            "tr1c": {"path": "/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"},
            "chanrg_train": {"path": "/root/rna_jepa_data/ss_data/jsonl/chanrg_train_train.jsonl"},
            "chanrg_validation": {"path": "/root/rna_jepa_data/ss_data/jsonl/chanrg_test.jsonl"},
        }}
        with self.assertRaises(AssertionError):
            validate_legal_inputs(summary)

    def test_exact_structure_does_not_imply_structure_clean(self):
        evidence = {"exact_structure_signature": "EXACT", "structure_similarity": "UNCHECKED"}
        self.assertNotEqual(evidence["exact_structure_signature"], evidence["structure_similarity"])
        self.assertEqual(evidence["structure_similarity"], "UNCHECKED")


if __name__ == "__main__":
    unittest.main()
