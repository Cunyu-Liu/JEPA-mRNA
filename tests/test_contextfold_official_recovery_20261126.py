#!/usr/bin/env python3
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
