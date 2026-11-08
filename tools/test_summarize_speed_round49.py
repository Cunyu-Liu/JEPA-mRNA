import unittest

from summarize_speed_round49 import ranking, validate_gpu


class SpeedSummaryValidationTests(unittest.TestCase):
    def valid_selected(self):
        return {"status": "PASS_STRUCTRFM_RAW_SPEED", "throughput_median_seq_per_s": 10.0,
                "protocol": {"n_raw": 79, "data_sha256": "abc", "device": "cuda",
                             "gpu_name": "NVIDIA GeForce RTX 4090", "batch_size": 1,
                             "warmup_sequences_per_repeat": 1, "repeats": 3, "cached_embeddings": False},
                "runs": [{"n": 79, "n_timed": 78}] * 3}

    def test_selected_full_raw_contract(self):
        self.assertEqual(validate_gpu(self.valid_selected(), "abc"), 10.0)

    def test_wrong_data_or_incomplete_repeats_fail(self):
        with self.assertRaises(AssertionError):
            validate_gpu(self.valid_selected(), "different")
        document = self.valid_selected()
        document["runs"][0]["n_timed"] = 77
        with self.assertRaises(AssertionError):
            validate_gpu(document, "abc")

    def test_ranking_is_not_seed_selection(self):
        rows = [{"model": "s0", "throughput_seq_per_s": 2.0},
                {"model": "s1", "throughput_seq_per_s": 3.0}]
        self.assertEqual(ranking(rows)[0]["model"], "s1")


if __name__ == "__main__":
    unittest.main()
