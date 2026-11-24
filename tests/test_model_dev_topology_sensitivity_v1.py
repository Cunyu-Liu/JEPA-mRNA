from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_model_dev_topology_sensitivity_v1 as B
import verify_model_dev_topology_sensitivity_v1 as V


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class TopologySensitivityTest(unittest.TestCase):
    def fixture(self, root):
        raw_path = root / "raw.jsonl"
        panel_path = root / "panel.jsonl"
        train_path = root / "train.txt"
        dev_path = root / "dev.txt"
        specifications = [
            ("train", "same", "AAAAAA", [(0, 5), (1, 4)], 1, "RF1"),
            ("train", "other", "CCCCCC", [(0, 5)], 2, "RF2"),
            ("dev", "same", "GGGGGG", [(0, 5), (1, 4)], 3, "RF1"),
            ("dev", "unique", "UUUUUUU", [(0, 6), (1, 5), (2, 4)], 4, None),
            ("dev", "same", "ACACAC", [(0, 5), (1, 4)], 5, "RF3"),
        ]
        raw_rows, panel_rows, train_ids, dev_ids = [], [], [], []
        for index, (assignment, name, sequence, pairs, cluster, family) in enumerate(specifications):
            sequence_id = f"bprna_tr1c:row{index:06d}:{name}"
            sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
            raw_rows.append({"name": name, "seq": sequence, "pairs": pairs})
            panel_rows.append({
                "sequence_id": sequence_id, "source_row_index": index, "name": name,
                "cluster_assignment": assignment, "cluster_id": cluster,
                "sequence_sha256": sequence_hash, "length": len(sequence), "n_pairs": len(pairs),
                "family_id": family, "family_status": "PASS" if family else "NOT_EXACT_IN_RFAM14.9",
            })
            (train_ids if assignment == "train" else dev_ids).append(sequence_id)
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
        panel_path.write_text("".join(json.dumps(row) + "\n" for row in panel_rows))
        train_path.write_text("".join(value + "\n" for value in train_ids))
        dev_path.write_text("".join(value + "\n" for value in dev_ids))
        registry_path = root / "registry.json"
        registry_path.write_text(json.dumps({
            "schema_version": "foldnar.model_dev_registry.v1", "frozen_utc": "synthetic",
            "assets": [{"dataset_id": "bprna_tr1c", "role": "legal_train_dev_source",
                        "source": str(raw_path), "sha256": identity(raw_path)["sha256"]}],
            "environment_contracts": {"panel_records": identity(panel_path)},
            "panels": [{"panel_id": "tr1c_cluster80_dev_v1", "split_unit": "existing 80% cluster",
                        "train_ids": identity(train_path), "dev_ids": identity(dev_path)}],
        }))
        return registry_path, train_ids, dev_ids

    def test_build_and_independent_raw_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, train_ids, dev_ids = self.fixture(root)
            out = root / "artifacts"
            manifest = B.build(registry, out)
            self.assertEqual(manifest["n_shared_topology_signatures"], 1)
            self.assertEqual(manifest["n_planned_train_records_with_shared_signature"], 1)
            self.assertEqual(manifest["scopes"]["raw"]["n_sequences"], 3)
            self.assertEqual(manifest["scopes"]["raw"]["n_pairs"], 7)
            self.assertEqual(manifest["scopes"]["raw"]["n_exact_topology_overlap_sequences"], 2)
            self.assertEqual(manifest["scopes"]["raw"]["n_exact_topology_overlap_pairs"], 4)
            self.assertEqual(manifest["scopes"]["own_clean"]["n_sequences"], 1)
            self.assertEqual(manifest["scopes"]["own_clean"]["n_pairs"], 3)
            self.assertEqual(manifest["scopes"]["own_clean"]["leakage_state"]["exact_loop_helix_topology_signature"], "NOT_DETECTED_UNKNOWN")
            self.assertEqual(manifest["scopes"]["own_clean"]["leakage_state"]["approximate_structural_similarity"], "UNCHECKED")
            self.assertEqual(manifest["scopes"]["matched_clean"], manifest["scopes"]["own_clean"] | {"scope": "matched_clean"})
            self.assertEqual(B.read_ids(out / "raw_cluster80_dev_ids.txt"), sorted(dev_ids))
            excluded = B.read_ids(out / "exact_topology_overlap_dev_ids.txt")
            self.assertEqual(excluded, sorted([dev_ids[0], dev_ids[2]]))
            concentration = json.loads((out / "concentration.json").read_text())
            self.assertEqual(concentration["raw"]["cluster80"]["n_groups"], 3)
            self.assertEqual(concentration["own_clean"]["family"]["n_groups"], 1)
            verification = V.verify(out / "sensitivity_manifest.json")
            self.assertEqual(verification["status"], "PASS")
            self.assertTrue(all(verification["checks"].values()))
            before = (out / "sensitivity_manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                B.build(registry, out)
            self.assertEqual((out / "sensitivity_manifest.json").read_bytes(), before)

    def test_frozen_identity_mismatch_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, _, dev_ids = self.fixture(root)
            dev_path = Path(json.loads(registry.read_text())["panels"][0]["dev_ids"]["path"])
            dev_path.write_text("unexpected\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                B.build(registry, root / "new-output")
            self.assertFalse((root / "new-output").exists())


if __name__ == "__main__":
    unittest.main()
