import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/recover_spot_rnaformer_v3_20261124.py"
SPEC = importlib.util.spec_from_file_location("recover_spot_rnaformer_v3_20261124", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairSetTest(unittest.TestCase):
    def test_zero_based_bounds_and_partner_conflicts(self):
        self.assertEqual(MODULE.pair_set([[0, 3], [1, 2]], 4), {(0, 3), (1, 2)})
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[-1, 2]], 4)
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 4]], 4)
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 3], [0, 2]], 4)

    def test_exact_pair_counts_without_span_filtering(self):
        rows = [{"name": "x", "seq": "AAAAA", "pairs": [[0, 1], [2, 3]]}]
        predictions = {"x": {"pairs": [[0, 1], [2, 4]]}}
        metrics, per_sequence = MODULE.strict_metrics(rows, predictions)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertEqual(metrics["micro_f1"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(per_sequence[0]["n_truth_pairs"], 2)

    def test_zero_pair_convention_and_micro_undefined(self):
        rows = [{"name": "empty", "seq": "AA", "pairs": []}]
        metrics, _ = MODULE.strict_metrics(rows, {"empty": {"pairs": []}})
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertIsNone(metrics["micro_f1"])
        self.assertEqual(metrics["coverage"], 1.0)

    def test_missing_prediction_is_not_imputed(self):
        rows = [{"name": "present", "seq": "AA", "pairs": []},
                {"name": "missing", "seq": "AA", "pairs": []}]
        with self.assertRaisesRegex(AssertionError, "MISSING_PREDICTION: missing"):
            MODULE.strict_metrics(rows, {"present": {"pairs": []}})


if __name__ == "__main__":
    unittest.main()
