import unittest

from compare_ripe_paired_round48 import common_rows


class CommonRowsTests(unittest.TestCase):
    def test_duplicate_and_blank_names_are_preserved(self):
        left = {"data_sha256": "x", "raw_n": 5,
                "per_sequence": [{"name": "", "f1": 0.1}, {"name": "", "f1": 0.2},
                                 {"name": "same", "f1": 0.3}, {"name": "same", "f1": 0.4}]}
        right = {"data_sha256": "x", "raw_n": 5,
                 "per_sequence": [{"name": "", "f1": 0.5}, {"name": "", "f1": 0.6},
                                  {"name": "same", "f1": 0.7}, {"name": "same", "f1": 0.8},
                                  {"name": "extra", "f1": 0.9}]}
        left_rows, right_rows, extra = common_rows(left, right)
        self.assertEqual(left_rows, left["per_sequence"])
        self.assertEqual(right_rows, right["per_sequence"][:4])
        self.assertEqual(extra, [("extra", 0)])

    def test_missing_supported_occurrence_is_rejected(self):
        left = {"data_sha256": "x", "raw_n": 2,
                "per_sequence": [{"name": "", "f1": 0.1}, {"name": "", "f1": 0.2}]}
        right = {"data_sha256": "x", "raw_n": 2, "per_sequence": [{"name": "", "f1": 0.3}]}
        with self.assertRaisesRegex(AssertionError, "lost supported sequences"):
            common_rows(left, right)

    def test_data_identity_is_not_replaced_by_positional_pairing(self):
        left = {"data_sha256": "left", "raw_n": 1, "per_sequence": [{"name": "", "f1": 0.1}]}
        right = {"data_sha256": "right", "raw_n": 1, "per_sequence": [{"name": "", "f1": 0.2}]}
        with self.assertRaises(AssertionError):
            common_rows(left, right)


if __name__ == "__main__":
    unittest.main()
