#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "recover_efold_family_baselines_v3_20261124.py"
SPEC = importlib.util.spec_from_file_location("baseline_recovery_efold_family_v3", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExactPairScoringTest(unittest.TestCase):
    def test_micro_macro_and_zero_pair_convention(self):
        rows = [
            {"name": "a", "seq": "AAA", "pairs": [[0, 2]]},
            {"name": "b", "seq": "AAAA", "pairs": []},
        ]
        predictions = {"a": {"pairs": [[0, 2]]}, "b": {"pairs": []}}
        metrics, per_sequence = MODULE.strict_metrics(rows, predictions)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(per_sequence[1]["f1"], 1.0)

    def test_false_positive_and_false_negative_are_retained(self):
        rows = [{"name": "a", "seq": "AAAA", "pairs": [[0, 3]]}]
        metrics, _ = MODULE.strict_metrics(rows, {"a": {"pairs": [[0, 2]]}})
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))
        self.assertEqual(metrics["micro_precision"], 0.0)
        self.assertEqual(metrics["micro_recall"], 0.0)
        self.assertEqual(metrics["micro_f1"], 0.0)

    def test_partner_conflict_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 3], [0, 2]], 4)

    def test_out_of_range_pair_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 4]], 4)

    def test_name_scope_mismatch_rejected(self):
        rows = [{"name": "a", "seq": "AAA", "pairs": []}]
        with self.assertRaises(AssertionError):
            MODULE.strict_metrics(rows, {"b": {"pairs": []}})

    def test_legacy_hash_uses_stable_length_group_order(self):
        rows = [
            {"name": "long", "seq": "AAAA", "pairs": []},
            {"name": "short", "seq": "AA", "pairs": []},
        ]
        predictions = {"long": {"pairs": [[0, 3]]}, "short": {"pairs": []}}
        legacy = [{"name": "short", "pairs": []}, {"name": "long", "pairs": [[0, 3]]}]
        import hashlib
        expected = hashlib.sha256(MODULE.canonical(legacy).encode()).hexdigest()
        self.assertEqual(MODULE.legacy_efold_prediction_sha256(rows, predictions), expected)

    def test_blocked_metrics_and_coverage_are_null(self):
        result = MODULE.unavailable_metrics("missing")
        self.assertIsNone(result["metrics"])
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["per_sequence_predictions"])


if __name__ == "__main__":
    unittest.main()
