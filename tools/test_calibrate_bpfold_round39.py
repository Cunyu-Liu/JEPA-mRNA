import unittest

import numpy as np

from calibrate_bpfold_round39 import flatten_probability


class BPfoldProbabilityTest(unittest.TestCase):
    def test_flatten_probability_preserves_gt_and_shape(self):
        row = {"seq": "AUGCAU", "pairs": [[0, 5]]}
        probability = np.full((6, 6), 0.25, dtype=np.float32)
        flat, labels = flatten_probability(row, probability)
        self.assertEqual(flat.dtype, np.float32)
        self.assertEqual(labels.sum(), 1)
        self.assertEqual(len(flat), len(labels))
        self.assertEqual(flat[labels == 1].tolist(), [0.25])

    def test_invalid_probability_is_rejected(self):
        row = {"seq": "AUGCAU", "pairs": [[0, 5]]}
        with self.assertRaises(AssertionError):
            flatten_probability(row, np.zeros((5, 5)))
        probability = np.full((6, 6), 0.25)
        probability[0, 5] = 1.5
        with self.assertRaises(AssertionError):
            flatten_probability(row, probability)

    def test_logits_are_converted_to_probabilities(self):
        from calibrate_bpfold_round39 import flatten_score
        row = {"seq": "AUGCAU", "pairs": [[0, 5]]}
        scores, probabilities, labels = flatten_score(row, np.full((6, 6), 2.0))
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(probabilities.dtype, np.float32)
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))
        self.assertEqual(labels.sum(), 1)
        self.assertAlmostEqual(float(probabilities[0]), 1 / (1 + np.exp(-2.0)), places=7)


if __name__ == "__main__":
    unittest.main()
