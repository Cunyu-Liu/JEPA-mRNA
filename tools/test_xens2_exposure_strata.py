import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_xens2_exposure_strata import metrics, normalize, history_differences


class ExposureStrataTest(unittest.TestCase):
    def test_micro_macro_counts(self):
        rows = [{"f1": 1.0, "n_pred": 1, "n_gt": 1}, {"f1": 0.0, "n_pred": 0, "n_gt": 2}]
        result = metrics(rows)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 2))
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertAlmostEqual(result["micro_f1"], 0.5)
        self.assertEqual(metrics([])["macro_f1"], None)

    def test_t_u_normalization(self):
        self.assertEqual(normalize("acgtu"), "ACGUU")

    def test_annotation_does_not_change_history(self):
        older = {"x": {"name": "x", "f1": 0.5, "n_pred": 1, "n_gt": 1}}
        rows = [{"name": "x", "f1": 0.5, "n_pred": 1, "n_gt": 1, "query_id": "Q1", "homology_detected": True}]
        self.assertEqual(history_differences(rows, older), 0)


if __name__ == "__main__":
    unittest.main()
