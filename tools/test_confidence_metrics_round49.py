import unittest

from confidence_metrics_round49 import correlation_with_ci, evaluate_structure_confidence
from scipy.stats import pearsonr


class ConfidenceMetricTests(unittest.TestCase):
    def test_pearson_spearman_ci_are_deterministic(self):
        rows = [{"structure_conf": value, "f1": f1}
                for value, f1 in [(0.1, 0.2), (0.3, 0.4), (0.5, 0.5), (0.7, 0.8), (0.9, 0.85)]]
        first = evaluate_structure_confidence(rows, [0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
        second = evaluate_structure_confidence(rows, [0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
        self.assertEqual({key: value for key, value in first.items() if key != "risk"},
                         {key: value for key, value in second.items() if key != "risk"})
        self.assertEqual(str(first["risk"]), str(second["risk"]))
        self.assertEqual(first["bootstrap"], 2000)
        self.assertEqual(first["bootstrap_seed"], 42)
        self.assertAlmostEqual(first["pearson"]["value"], 0.9814954576223635, places=12)
        self.assertEqual(len(first["pearson"]["ci95"]), 2)
        self.assertEqual(len(first["spearman"]["ci95"]), 2)

    def test_constant_confidence_has_no_correlation(self):
        result = correlation_with_ci([1.0, 1.0, 1.0], [0.2, 0.4, 0.8], pearsonr, 2000)
        self.assertIsNone(result["value"])
        self.assertIsNone(result["ci95"])

    def test_empty_risk_bucket_remains_null(self):
        rows = [{"structure_conf": 0.1, "f1": 0.2}, {"structure_conf": 0.2, "f1": 0.4}]
        result = evaluate_structure_confidence(rows, [0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
        self.assertIsNone(result["aurc"])
        self.assertTrue(any(str(value) == "nan" for value in result["risk"]))
        with self.assertRaises(ValueError):
            evaluate_structure_confidence(rows, [0.0, 0.5, 0.8])

    def test_bootstrap_count_is_frozen(self):
        with self.assertRaises(ValueError):
            correlation_with_ci([0.1, 0.2], [0.2, 0.4], pearsonr, 100)


if __name__ == "__main__":
    unittest.main()
