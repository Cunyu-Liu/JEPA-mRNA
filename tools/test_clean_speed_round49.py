import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path

from clean_speed_round49 import add_clean_speed_args, validate_clean_speed_rows


class CleanSpeedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "bprna_ts0.jsonl"
        path.write_text('{"seq":"ACGU"}\n{"seq":"UGCA"}\n')
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", type=Path)
        parser.add_argument("--repeats", type=int, default=3)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--clean-audit", type=Path)
        add_clean_speed_args(parser)
        self.args = parser.parse_args(["--data", str(path), "--clean-speed-protocol",
                                       "--expected-data-sha256", hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "--expected-count", "2"])

    def tearDown(self):
        self.temporary.cleanup()

    def test_registered_full_clean_set(self):
        self.assertTrue(validate_clean_speed_rows([{"seq": "ACGU"}, {"seq": "UGCA"}], self.args))

    def test_count_hash_and_incomplete_repeats_fail_closed(self):
        rows = [{"seq": "ACGU"}, {"seq": "UGCA"}]
        self.args.expected_count = 3
        with self.assertRaises(ValueError):
            validate_clean_speed_rows(rows, self.args)
        self.args.expected_count = 2
        self.args.expected_data_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            validate_clean_speed_rows(rows, self.args)
        self.args.expected_data_sha256 = hashlib.sha256(self.args.data.read_bytes()).hexdigest()
        self.args.repeats = 1
        with self.assertRaises(ValueError):
            validate_clean_speed_rows(rows, self.args)

    def test_smoke_subset_and_other_audit_cannot_be_promoted(self):
        self.args.smoke = True
        with self.assertRaises(ValueError):
            validate_clean_speed_rows([{"seq": "ACGU"}, {"seq": "UGCA"}], self.args)


if __name__ == "__main__":
    unittest.main()
