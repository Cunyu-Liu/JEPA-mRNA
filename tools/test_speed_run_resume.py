import json
import tempfile
import unittest
from pathlib import Path

from speed_end_to_end_4090 import load_partial_runs, prediction_digest, save_partial_runs


class RunResumeTest(unittest.TestCase):
    def setUp(self):
        self.records = [{"name": "a"}, {"name": "b"}]
        self.predictions = [[[0, 3]], []]
        self.identity = {"repeats": 3, "data_sha256": "data", "code_sha256": "code"}
        self.runs = [{"repeat": 0, "predictions_sha256": prediction_digest(self.records, self.predictions)}]

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            save_partial_runs(path, self.identity, self.runs, [self.predictions])
            runs, predictions = load_partial_runs(path, self.identity, self.records)
            self.assertEqual(runs, self.runs)
            self.assertEqual(predictions, [self.predictions])
            self.assertEqual(json.loads(path.read_text())["status"], "PARTIAL_RUNS_NOT_SCIENTIFIC_RESULT")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_missing_is_clean_start(self):
        self.assertEqual(load_partial_runs(Path("missing"), self.identity, self.records), ([], []))

    def test_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            save_partial_runs(path, self.identity, self.runs, [self.predictions])
            with self.assertRaisesRegex(AssertionError, "identity mismatch"):
                load_partial_runs(path, {**self.identity, "data_sha256": "other"}, self.records)

    def test_corrupt_prediction_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            save_partial_runs(path, self.identity, self.runs, [self.predictions])
            with self.assertRaises(AssertionError):
                load_partial_runs(path, self.identity, self.records[:1])


if __name__ == "__main__":
    unittest.main()
