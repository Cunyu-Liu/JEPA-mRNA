#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load_module("contextfold_official_recovery_20261126")
sys.path.insert(0, str(TOOLS))
launcher = load_module("run_contextfold_official_bprna_new")
verifier = load_module("verify_contextfold_official_recovery_20261126")
baseline = load_module("foldnar_baseline_contract")


class NativeContractTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"name": "a", "seq": "AAGGCCUUGGGGAAGGCCUU", "pairs": []},
            {"name": "b", "seq": "ACGU", "pairs": []},
        ]

    def test_native_input_never_writes_reference_structures(self):
        text = recovery.native_input(self.rows)
        self.assertEqual(text, ">a\nAAGGCCUUGGGGAAGGCCUU\n>b\nACGU\n")
        self.assertNotIn("(", text)

    def test_native_output_to_zero_based_pairs(self):
        text = ">a\nAAGGCCUUGGGGAAGGCCUU\n(((((..........)))))\n>b\nACGU\n....\n"
        predictions = recovery.parse_native_output(text, self.rows)
        self.assertEqual(predictions[0]["pred_pairs"], [[0, 19], [1, 18], [2, 17], [3, 16], [4, 15]])
        self.assertEqual(predictions[1]["pred_pairs"], [])

    def test_partial_summary_or_reordered_output_rejected(self):
        with self.assertRaises(recovery.ContractError):
            recovery.parse_native_output(">a\nAAGGCCUUGGGGAAGGCCUU\n....................\n", self.rows)
        with self.assertRaises(recovery.ContractError):
            recovery.parse_native_output(">b\nACGU\n....\n>a\nAAGGCCUUGGGGAAGGCCUU\n....................\n", self.rows)

    def test_invalid_structures_rejected(self):
        for structure in ("(....", "....)", "(((....", "[...]"):
            with self.subTest(structure=structure), self.assertRaises(recovery.ContractError):
                recovery.dotbracket_pairs(structure, "ACGU")

    def test_crossing_and_partner_conflicts_rejected(self):
        predictions = [
            {"name": "a", "seq": self.rows[0]["seq"], "pred_pairs": [[0, 10], [5, 15]]},
            {"name": "b", "seq": "ACGU", "pred_pairs": []},
        ]
        with self.assertRaisesRegex(recovery.ContractError, "crossing"):
            recovery.validate_prediction_rows(self.rows, predictions)
        predictions[0]["pred_pairs"] = [[0, 10], [0, 11]]
        with self.assertRaisesRegex(recovery.ContractError, "at most one"):
            recovery.validate_prediction_rows(self.rows, predictions)


class DatasetContractTest(unittest.TestCase):
    def test_real_bprna_new_identity(self):
        rows = recovery.load_dataset()
        self.assertEqual(len(rows), 5388)
        self.assertEqual(len({row["name"] for row in rows}), 5388)
        self.assertEqual(min(map(lambda row: len(row["seq"]), rows)), 33)
        self.assertEqual(max(map(lambda row: len(row["seq"]), rows)), 489)
        self.assertEqual(sum(len(row["pairs"]) for row in rows), 135184)

    def test_small_loader_is_explicitly_nonfull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in [
                {"name": "a", "seq": "ACGU"}, {"name": "b", "seq": "AAAA"},
            ]))
            self.assertEqual(len(recovery.load_dataset(path, require_full=False)), 2)
            with self.assertRaises(recovery.ContractError):
                recovery.load_dataset(path)


class RecoveryManifestTest(unittest.TestCase):
    def test_blocked_manifest_never_fakes_assets_or_predictions(self):
        path = Path(__file__).parents[1] / "manifests/foldnar_v2/contextfold_official_recovery_20261126.json"
        document = json.loads(path.read_text())
        self.assertFalse(document["full_ready"])
        self.assertEqual(document["implementation"]["archive"]["sha256"], None)
        self.assertEqual(document["trained_parameters"]["sha256"], None)
        self.assertIsNone(document["bprna_new_5388_contract"]["predictions"])
        self.assertEqual(document["bprna_new_5388_contract"]["n_sequences"], 5388)
        self.assertFalse(document["execution"]["gpu_tasks_started_or_modified"])
        self.assertEqual(document["official_identity"]["archive_url"], recovery.OFFICIAL_ARCHIVE_URL)


class LauncherTest(unittest.TestCase):
    def test_missing_official_implementation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"implementation": {"status": "BLOCKED"}}))
            with self.assertRaisesRegex(launcher.PreflightError, "fail-closed"):
                launcher.preflight(root, manifest, root / "dataset.jsonl")

    def test_asset_and_output_cannot_escape_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assets"
            root.mkdir()
            with self.assertRaises(launcher.PreflightError):
                launcher.inside_root(Path(directory) / "prediction.jsonl", root)

    def test_real_cli_makes_no_prediction_file_when_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction.jsonl"
            command = [
                sys.executable,
                str(TOOLS / "run_contextfold_official_bprna_new.py"),
                "--output", str(output),
                "--asset-root", directory,
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("PREFLIGHT_BLOCKED", completed.stderr)
            self.assertFalse(output.exists())


class IndependentVerificationTest(unittest.TestCase):
    def test_original_evidence_dataset_and_registry_pass(self):
        report = verifier.verify()
        self.assertEqual(report["status"], "PASS_BLOCKED_OFFICIAL_RECOVERY_CONTRACT")
        self.assertEqual(report["n_dataset_rows"], 5388)
        self.assertFalse(report["full_ready"])
        self.assertIsNone(report["predictions"])

    def test_registry_and_generator_keep_the_same_identity(self):
        registry = json.loads((Path(__file__).parents[1] / "manifests/foldnar_v1/baseline_registry.json").read_text())
        row = next(row for row in registry["rows"] if row["baseline_id"] == "contextfold")
        spec = next(spec for spec in baseline._specs() if spec["baseline_id"] == "contextfold")
        training = baseline._default_training("contextfold")
        self.assertEqual(row["category"], "STRUCTURED_DECODER")
        self.assertEqual(row["version_protocol"], spec["version_protocol"])
        self.assertEqual(row["status"], spec["status"])
        self.assertEqual(row["training_data"]["status"], training["status"])
        self.assertEqual(row["training_data"]["external_training_corpora"], training["external_training_corpora"])
        self.assertEqual(row["prediction_ids"], [])
        self.assertEqual(row["n_frozen_prediction_rows"], 0)
        self.assertEqual(row["n_summary_only_rows"], 0)


if __name__ == "__main__":
    unittest.main()
