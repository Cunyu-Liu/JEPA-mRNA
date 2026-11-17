import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from extract_interface_features import structure_statistics
from train_interface_baseline import metrics


class InterfaceMetricTest(unittest.TestCase):
    def test_binary_macro_is_not_positive_f1(self):
        predictions = [{"probability": np.array([0.9, 0.8, 0.2, 0.1]),
                        "label": np.array([1.0, 0.0, 1.0, 0.0])}]
        result = metrics(predictions, 0.15)
        self.assertAlmostEqual(result["micro_f1"], 0.8)
        self.assertAlmostEqual(result["macro_f1_binary"], (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(result["micro_auprc"], (1 + 2 / 3) / 2)

    def test_auprc_is_global_not_columnwise(self):
        predictions = [{"probability": np.array([0.9, 0.7]), "label": np.array([1.0, 0.0])},
                       {"probability": np.array([0.8, 0.75, 0.65]), "label": np.array([1.0, 0.0, 1.0])}]
        result = metrics(predictions, 0.5)
        self.assertAlmostEqual(result["micro_auprc"], (1 + 1 + 3 / 5) / 3)
        self.assertAlmostEqual(result["macro_auprc_positive_chains"], (1 + 5 / 6) / 2)
        self.assertEqual(result["n_auprc_chains"], 2)

    def test_positive_only_chain_has_no_undefined_auprc(self):
        result = metrics([{"probability": np.array([0.6, 0.4]), "label": np.array([1.0, 1.0])}], 0.5)
        self.assertEqual(result["micro_auprc"], 1.0)
        self.assertEqual(result["macro_auprc_positive_chains"], 1.0)

    def test_structure_features_finite_for_empty_and_single_residue(self):
        for sequence in ("A", "AA", "AUGCUAUGCA"):
            scores = torch.arange(len(sequence) ** 2, dtype=torch.float32).reshape(len(sequence), len(sequence))
            features = structure_statistics(scores, sequence)
            self.assertEqual(features.shape, (len(sequence), 8))
            self.assertTrue(torch.isfinite(features).all())
            self.assertTrue(torch.all(features[:, [0, 1, 2, 3, 5, 6, 7]] >= 0))
            self.assertTrue(torch.all(features[:, 4].abs() <= 1))


if __name__ == "__main__":
    unittest.main()
