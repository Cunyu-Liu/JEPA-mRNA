#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load_module("debfold_official_recovery_20261126")
launcher = load_module("run_debfold_bprna_new_gpu_20261126")


class CTParserTest(unittest.TestCase):
    def test_energy_header_and_reciprocal_pairs(self):
        row = recovery.parse_ct("4 Energy = 0 sample\n1 A 0 2 4 1\n2 U 1 3 0 2\n3 G 2 4 0 3\n4 C 3 5 1 4\n")
        self.assertEqual(row["seq"], "AUGC")
        self.assertEqual(row["pairs"], [(0, 3)])

    def test_nonreciprocal_pair_rejected(self):
        with self.assertRaises(ValueError):
            recovery.parse_ct("2 sample\n1 A 0 2 2 1\n2 C 1 3 0 2\n")

    def test_invalid_alphabet_rejected(self):
        with self.assertRaises(ValueError):
            recovery.parse_ct("1 sample\n1 X 0 2 0 1\n")


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"name": "a", "seq": "AUGC", "pairs": [[0, 3]]},
            {"name": "b", "seq": "AAAA", "pairs": []},
        ]

    def test_complete_per_sequence_output(self):
        predictions = [
            {"name": "a", "seq": "AUGC", "pred_pairs": [[0, 3]]},
            {"name": "b", "seq": "AAAA", "pred_pairs": []},
        ]
        self.assertEqual(set(recovery.validate_prediction_rows(self.rows, predictions)), {"a", "b"})

    def test_summary_only_or_partial_output_rejected(self):
        with self.assertRaises(ValueError):
            recovery.validate_prediction_rows(self.rows, [{"name": "a", "seq": "AUGC", "pred_pairs": [[0, 3]]}])

    def test_sequence_identity_required(self):
        predictions = [
            {"name": "a", "seq": "AAAA", "pred_pairs": []},
            {"name": "b", "seq": "AAAA", "pred_pairs": []},
        ]
        with self.assertRaises(ValueError):
            recovery.validate_prediction_rows(self.rows, predictions)

    def test_partner_and_duplicate_rules(self):
        invalid_pairs = ([[3, 0]], [[0, 4]], [[0, 3], [0, 3]])
        for pairs in invalid_pairs:
            predictions = [
                {"name": "a", "seq": "AUGC", "pred_pairs": pairs},
                {"name": "b", "seq": "AAAA", "pred_pairs": []},
            ]
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                recovery.validate_prediction_rows(self.rows, predictions)

    def test_small_dataset_loader_is_explicitly_nonfull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
            self.assertEqual(len(recovery.load_dataset(path, require_full=False)), 2)
            with self.assertRaises(ValueError):
                recovery.load_dataset(path)


class UnifiedRegistryTest(unittest.TestCase):
    def test_training_archive_update_preserves_blocked_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / "manifests/foldnar_v1/baseline_registry.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"rows": [{
                "baseline_id": "debfold",
                "training_data": {"status": "BLOCKED_PROVENANCE"},
                "prediction_ids": [],
                "n_frozen_prediction_rows": 0,
            }]}))
            recovery.update_unified_registry(repo)
            row = json.loads(path.read_text())["rows"][0]
            self.assertEqual(row["training_data"]["archive_sha256"], recovery.UNIFIED_TRAINING["archive_sha256"])
            self.assertEqual(row["prediction_ids"], [])
            self.assertEqual(row["n_frozen_prediction_rows"], 0)


class LauncherTest(unittest.TestCase):
    def test_missing_official_adapter_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps({"name": "a", "seq": "AUGC", "pairs": []}) + "\n")
            with unittest.mock.patch.object(launcher, "load_dataset", return_value=[{"name": "a", "seq": "AUGC"}]):
                with self.assertRaises(launcher.PreflightError):
                    launcher.preflight(root / "absent.json", root, dataset)

    def test_over_512_requires_official_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter.json"
            adapter.write_text(json.dumps({
                "status": "READY_OFFICIAL_ADAPTER_VERIFIED",
                "official_repository": "https://github.com/cobisLab/DEBFold",
                "official_commit": recovery.OFFICIAL_COMMIT,
            }))
            rows = [{"name": "long", "seq": "A" * 513}]
            with unittest.mock.patch.object(launcher, "load_dataset", return_value=rows):
                with self.assertRaisesRegex(launcher.PreflightError, "over 512"):
                    launcher.preflight(adapter, root, root / "dataset.jsonl")

    def test_official_asset_must_be_under_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assets"
            outside = Path(directory) / "weights.pt"
            root.mkdir()
            outside.write_bytes(b"not-a-checkpoint")
            with self.assertRaises(launcher.PreflightError):
                launcher.inside_root(outside, root)


if __name__ == "__main__":
    unittest.main()
