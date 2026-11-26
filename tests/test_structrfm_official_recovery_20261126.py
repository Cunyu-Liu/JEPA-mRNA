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


recovery = load_module("structrfm_official_recovery_20261126")
launcher = load_module("run_structrfm_bprna_new_gpu_20261126")
differential = load_module("structrfm_interface_differential_20261126")


class PairContractTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"name": "a", "seq": "AUGC", "pairs": [[0, 3]]},
            {"name": "b", "seq": "AAAA", "pairs": []},
        ]

    def test_reciprocal_one_based_connects(self):
        self.assertEqual(recovery.pairs_from_connects([4, 0, 0, 1], "AUGC"), [(0, 3)])

    def test_nonreciprocal_connects_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-reciprocal"):
            recovery.pairs_from_connects([4, 0, 0, 0], "AUGC")

    def test_complete_ordered_prediction_output(self):
        predictions = [{"name": "a", "seq": "AUGC", "connects": [4, 0, 0, 1]}, {"name": "b", "seq": "AAAA", "pred_pairs": []}]
        output = recovery.validate_prediction_rows(self.rows, predictions)
        self.assertEqual(output[0]["pred_pairs"], [[0, 3]])
        self.assertEqual(output[1]["pred_pairs"], [])

    def test_partial_or_summary_only_output_rejected(self):
        with self.assertRaises(ValueError):
            recovery.validate_prediction_rows(self.rows, [{"name": "a", "seq": "AUGC", "pred_pairs": [[0, 3]]}])
        with self.assertRaises(ValueError):
            recovery.validate_prediction_rows(self.rows, [{"micro_f1": 0.5}, {"micro_f1": 0.5}])

    def test_sequence_order_and_partner_conflicts_rejected(self):
        invalid = [
            [{"name": "b", "seq": "AAAA", "pred_pairs": []}, {"name": "a", "seq": "AUGC", "pred_pairs": [[0, 3]]}],
            [{"name": "a", "seq": "AAAA", "pred_pairs": [[0, 3]]}, {"name": "b", "seq": "AAAA", "pred_pairs": []}],
            [{"name": "a", "seq": "AUGC", "pred_pairs": [[0, 3], [0, 2]]}, {"name": "b", "seq": "AAAA", "pred_pairs": []}],
        ]
        for predictions in invalid:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                recovery.validate_prediction_rows(self.rows, predictions)

    def test_small_dataset_is_explicitly_nonfull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
            self.assertEqual(len(recovery.load_dataset(path, require_full=False)), 2)
            with self.assertRaises(ValueError):
                recovery.load_dataset(path)


class OfficialIdentityTest(unittest.TestCase):
    def test_real_bprna_new_contract(self):
        rows = recovery.load_dataset()
        self.assertEqual(len(rows), 5388)
        self.assertEqual(sum(len(row["pairs"]) for row in rows), 135184)
        self.assertEqual(min(len(row["seq"]) for row in rows), 33)
        self.assertEqual(max(len(row["seq"]) for row in rows), 489)

    def test_checkpoint_bytes_and_cpu_layout(self):
        report = recovery.inspect_checkpoint()
        self.assertEqual(report["sha256"], recovery.ASSET_SHA256)
        self.assertEqual(report["bytes"], recovery.ASSET_BYTES)
        self.assertEqual(report["load_device"], "cpu")
        self.assertGreater(report["state_dicts"]["model"]["n_tensors"], 0)
        self.assertGreater(report["state_dicts"]["pretrained_model"]["n_tensors"], 0)

    def test_checkpoint_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / recovery.ASSET_NAME
            path.write_bytes(b"not-an-official-checkpoint")
            with self.assertRaisesRegex(ValueError, "byte count mismatch"):
                recovery.validate_checkpoint(path)

    def test_official_source_and_release_commits(self):
        report = recovery.validate_source()
        self.assertEqual(report["release_commit"], recovery.RELEASE_COMMIT)
        self.assertEqual(report["inference_commit"], recovery.INFERENCE_COMMIT)
        self.assertEqual(report["inference_package_version"], "0.0.9")

    def test_historical_weight_missing_is_not_a_match(self):
        report = recovery.historical_weight_audit()
        self.assertIsNone(report["matches_official"])
        self.assertEqual(report["status"], "MISSING_HISTORICAL_WEIGHT_BYTES")


class DifferentialContractTest(unittest.TestCase):
    def test_cases_cover_synthetic_and_real_bprna_new_lengths(self):
        rows = differential.cases(recovery.DEFAULT_SOURCE)
        self.assertEqual(len(rows), 18)
        real = [row for row in rows if "dataset_name" in row]
        self.assertEqual(len(real), 6)
        self.assertEqual(min(len(row["sequence"]) for row in real), 33)
        self.assertEqual(max(len(row["sequence"]) for row in real), 489)


class LauncherTest(unittest.TestCase):
    def test_verified_interface_and_runtime_preflight(self):
        report = launcher.runtime_preflight()
        self.assertEqual(report["torch_cuda"], "12.8")
        self.assertIn("sm_120", report["torch_arch_list"])
        self.assertEqual(report["interface"]["sha256"], launcher.INTERFACE_SHA256)
        self.assertEqual(report["interface"]["differential_cases"], 18)

    def test_unverified_interface_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(launcher.PreflightError, "interface missing|Missing official cp38"):
                launcher.load_verified_interface(Path(directory))

    def test_missing_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(launcher.PreflightError):
                launcher.preflight(root=Path(directory), require_runtime=False)

    def test_execute_requires_runtime_and_never_falls_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(launcher.PreflightError, "Runtime preflight"):
                launcher.execute({"runtime": None, "root": root}, root / "predictions/out.jsonl")


class ManifestTest(unittest.TestCase):
    def test_manifest_never_fakes_new_predictions(self):
        path = Path(__file__).parents[1] / "manifests/foldnar_v2/structrfm_official_recovery_20261126.json"
        document = json.loads(path.read_text())
        self.assertFalse(document["full_ready"])
        self.assertTrue(document["asset_ready"])
        self.assertIsNone(document["bprna_new_5388_contract"]["predictions"])
        self.assertIsNone(document["bprna_new_5388_contract"]["metrics"])
        self.assertFalse(document["model_identity"]["ensemble"])
        self.assertEqual(document["model_identity"]["n_checkpoints"], 1)
        self.assertFalse(document["historical_artifacts"]["eligible_as_new_official_prediction"])
        self.assertTrue(document["gpu_execution"]["performed"])
        self.assertFalse(document["gpu_execution"]["full_dataset_inference_performed"])

    def test_runtime_manifest_is_ready_without_faking_full_predictions(self):
        root = Path(__file__).parents[1]
        document = json.loads((root / "manifests/foldnar_v2/structrfm_runtime_compatibility_20261126.json").read_text())
        self.assertTrue(document["runtime_ready"])
        self.assertFalse(document["full_ready"])
        self.assertIsNone(document["predictions"])
        self.assertIsNone(document["metrics"])
        self.assertFalse(document["gpu_readiness"]["full_dataset_inference_performed"])
        self.assertEqual(document["differential"]["n_cases"], 18)
        original = json.loads((root / "manifests/foldnar_v2/structrfm_official_recovery_20261126.json").read_text())
        self.assertTrue(original["runtime_ready"])
        self.assertFalse(original["gpu_execution"]["full_dataset_inference_performed"])

    def test_registry_blocks_unlinked_historical_prediction(self):
        root = Path(__file__).parents[1]
        baseline_doc = json.loads((root / "manifests/foldnar_v1/baseline_registry.json").read_text())
        baseline = next(row for row in baseline_doc["rows"] if row["baseline_id"] == "structrfm")
        self.assertEqual(baseline["official_weights"]["sha256"], recovery.ASSET_SHA256)
        self.assertEqual(baseline["n_frozen_prediction_rows"], 0)
        self.assertEqual(baseline["n_blocked_prediction_rows"], 1)
        prediction_doc = json.loads((root / "manifests/foldnar_v1/prediction_registry.json").read_text())
        prediction = next(row for row in prediction_doc["rows"] if row["prediction_id"] == "structrfm/bprna_new")
        self.assertEqual(prediction["status"], "BLOCKED")
        self.assertFalse(prediction["official_checkpoint_link"]["eligible_as_verified_official"])


if __name__ == "__main__":
    unittest.main()
