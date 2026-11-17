import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from combine_pretrain_exact_overlap import matched_sequences


class CombineExactTest(unittest.TestCase):
    def test_duplicate_rows_count_once(self):
        matched, n_rows = matched_sequences([("a", "ACGU"), ("a", "ACGU"), ("b", "UUUU")], ["a", "a"])
        self.assertEqual(matched, {"ACGU"})
        self.assertEqual(n_rows, 2)

    def test_name_collision_rejected(self):
        with self.assertRaises(AssertionError):
            matched_sequences([("a", "ACGU"), ("a", "UUUU")], ["a"])

    def test_name_collision_resolved_by_source_sequence(self):
        rows = [("a", "ACGU"), ("a", "UUUU")]
        matched, n_rows = matched_sequences(rows, ["a", "a"], {"ACGU"})
        self.assertEqual(matched, {"ACGU"})
        self.assertEqual(n_rows, 1)


if __name__ == "__main__":
    unittest.main()
