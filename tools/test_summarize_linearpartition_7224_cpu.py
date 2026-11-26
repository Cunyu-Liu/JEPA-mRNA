import unittest

from summarize_linearpartition_7224_cpu import pool_recall


class PilotSummaryTests(unittest.TestCase):
    def test_pool_recall_preserves_counts_and_null_empty_buckets(self):
        def chain(reference, tp):
            values = {name: {"reference": 0, "true_positive": 0, "recall": None}
                      for name in ("overall", "0-49", "50-199", "200-999", ">=1000")}
            values["overall"] = {"reference": reference, "true_positive": tp,
                                 "recall": tp / reference}
            values["50-199"] = dict(values["overall"])
            return {"recomputed": {"bpp_candidate_pair_recall": values}}
        result = pool_recall([chain(2, 1), chain(3, 2)], "bpp_candidate_pair_recall")
        self.assertEqual(result["overall"], {"reference": 5, "true_positive": 3, "recall": 0.6})
        self.assertIsNone(result[">=1000"]["recall"])


if __name__ == "__main__":
    unittest.main()
