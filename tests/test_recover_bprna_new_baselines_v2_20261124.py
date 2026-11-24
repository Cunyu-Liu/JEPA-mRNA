#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "recover_bprna_new_baselines_v2_20261124.py"
SPEC = importlib.util.spec_from_file_location("baseline_recovery_v2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairScoringTest(unittest.TestCase):
    def test_strict_raw_metrics_and_coverage(self):
        rows = [
            {"name": "a", "seq": "AAA", "pairs": [[0, 2]]},
            {"name": "b", "seq": "AAAA", "pairs": []},
        ]
        predictions = {
            "a": {"pairs": [[0, 2]]},
            "b": {"pairs": []},
        }
        metrics, per_sequence = MODULE.strict_metrics(rows, predictions)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(per_sequence[1]["f1"], 1.0)

    def test_errors_are_not_silently_dropped(self):
        rows = [{"name": "a", "seq": "AAAA", "pairs": [[0, 3]]}]
        predictions = {"a": {"pairs": [[0, 2]]}}
        metrics, _ = MODULE.strict_metrics(rows, predictions)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))
        self.assertEqual(metrics["micro_precision"], 0.0)
        self.assertEqual(metrics["micro_recall"], 0.0)
        self.assertEqual(metrics["micro_f1"], 0.0)

    def test_partner_conflict_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 3], [0, 2]], 4)

    def test_bpseq_one_based_to_zero_based(self):
        sequence, pairs = MODULE.parse_bpseq(b"1 A 4\n2 U 0\n3 G 0\n4 C 1\n")
        self.assertEqual(sequence, "AUGC")
        self.assertEqual(pairs, {(0, 3)})

    def test_bpseq_invalid_index_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.parse_bpseq(b"1 A 5\n2 U 0\n3 G 0\n4 C 1\n")


if __name__ == "__main__":
    unittest.main()
