import unittest

from tools.verify_prior_alignment_science import clean_pairs, counts, prf, stems


class PriorAlignmentScienceTests(unittest.TestCase):
    def test_prf_zero_denominator_semantics(self):
        self.assertEqual(prf(0, 0, 0), {"tp": 0, "fp": 0, "fn": 0,
                                        "precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(prf(0, 2, 0)["recall"], 1.0)
        self.assertEqual(prf(0, 0, 2)["precision"], 0.0)

    def test_pair_cleaning_rejects_crossing_and_conflict(self):
        with self.assertRaisesRegex(ValueError, "crossing"):
            clean_pairs([(0, 10), (2, 12)])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            clean_pairs([(0, 10), (0, 12)])

    def test_exact_stem_partition(self):
        pairs = [(2, 20), (3, 19), (4, 18), (8, 12)]
        self.assertEqual(stems(pairs), [((2, 20), (3, 19), (4, 18)), ((8, 12),)])
        self.assertEqual(counts(stems(pairs), stems([(2, 20), (3, 19), (8, 12)])),
                         {"tp": 1, "fp": 1, "fn": 1})

    def test_pair_counts(self):
        self.assertEqual(counts([(0, 9), (1, 8)], [(0, 9), (2, 7)]),
                         {"tp": 1, "fp": 1, "fn": 1})


if __name__ == "__main__":
    unittest.main()
