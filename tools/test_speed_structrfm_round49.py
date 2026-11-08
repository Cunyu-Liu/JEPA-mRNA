import unittest

from speed_structrfm_round49 import prediction_digest, summarize_runs


class SpeedSummaryTests(unittest.TestCase):
    def test_median_and_complete_raw_coverage(self):
        runs = [{"n": 79, "n_timed": 78, "throughput_seq_per_s": value,
                 "strict_micro_f1": 0.5, "peak_memory_mib": 1000 + value}
                for value in [2.0, 1.0, 3.0]]
        result = summarize_runs(runs)
        self.assertEqual(result["throughput_median_seq_per_s"], 2.0)
        self.assertEqual(result["peak_memory_mib"], 1003.0)
        self.assertEqual(result["f1_runs"], [0.5, 0.5, 0.5])

    def test_short_or_incomplete_runs_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize_runs([])
        with self.assertRaises(ValueError):
            summarize_runs([{"n": 79, "n_timed": 77, "throughput_seq_per_s": 1,
                             "strict_micro_f1": 0.5, "peak_memory_mib": 1}] * 3)

    def test_prediction_digest_binds_names_order_and_pairs(self):
        rows = [{"name": "a"}, {"name": "b"}]
        predictions = [[[0, 5]], []]
        digest = prediction_digest(rows, predictions)
        self.assertEqual(digest, prediction_digest(rows, predictions))
        self.assertNotEqual(digest, prediction_digest(rows[::-1], predictions))
        self.assertNotEqual(digest, prediction_digest(rows, [predictions[1], predictions[0]]))


if __name__ == "__main__":
    unittest.main()
