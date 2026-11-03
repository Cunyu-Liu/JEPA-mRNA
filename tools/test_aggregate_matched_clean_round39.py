import unittest

from aggregate_matched_clean_round39 import canonical_pairs, matched_group, matched_pairs, native_limit, score_entries, scoped_entries


class MatchedCleanTest(unittest.TestCase):
    def entry(self, sequence="AUGC", truth=((0, 3),), prediction=((0, 3),)):
        return {"sequence": sequence, "truth": set(truth), "prediction": set(prediction)}

    def test_canonical_pairs_scope(self):
        self.assertEqual(canonical_pairs([(3, 0), (1, 2)], 4), {(0, 3), (1, 2)})
        self.assertEqual(canonical_pairs([(3, 0), (1, 2)], 2), set())
        self.assertEqual(canonical_pairs([(3, 0), (1, 2)], 3), {(1, 2)})
        with self.assertRaises(AssertionError):
            canonical_pairs([(1, 1)])

    def test_score_and_matched_intersection(self):
        first = {"keep": self.entry(), "drop": self.entry()}
        second = {"keep": self.entry(prediction=((0, 1),)), "other": self.entry()}
        result = matched_group({"a": first, "b": second})
        self.assertEqual(result["a"], {"n": 1, "tp": 1, "fp": 0, "fn": 0, "f1": 1.0})
        self.assertEqual(result["b"], {"n": 1, "tp": 0, "fp": 1, "fn": 1, "f1": 0.0})
        pairwise = matched_pairs({"a": first, "b": second})
        self.assertEqual(pairwise["a|b"]["n"], 1)
        self.assertEqual(pairwise["a|b"]["systems"], result)

    def test_mismatch_is_fail_closed(self):
        with self.assertRaisesRegex(AssertionError, "sequence mismatch"):
            matched_group({"a": {"x": self.entry()}, "b": {"x": self.entry(sequence="AAAA")}})
        with self.assertRaisesRegex(AssertionError, "truth mismatch"):
            matched_group({"a": {"x": self.entry()}, "b": {"x": self.entry(truth=((0, 2),))}})

    def test_empty_intersection_is_not_a_win(self):
        result = matched_group({"a": {"x": self.entry()}, "b": {"y": self.entry()}})
        self.assertEqual(result["a"], {"n": 0, "status": "EMPTY_INTERSECTION"})
        self.assertNotIn("f1", result["a"])

    def test_native_limit(self):
        self.assertEqual(native_limit({"sequence_core_max_length": 598}, 598), 598)
        self.assertEqual(native_limit({"input_max_length_with_special_tokens": 1024},
                                      "input_max_length_with_special_tokens"), 1022)
        self.assertIsNone(native_limit({}, None))

    def test_length_excluded_scope(self):
        entries = {"short": self.entry(sequence="A" * 600), "long": self.entry(sequence="A" * 601)}
        value = {"scope": "full_length", "entries": entries}
        self.assertEqual(set(scoped_entries(value, "len_le600")), {"short"})
        self.assertEqual(scoped_entries(value, "full_length"), entries)
        self.assertIsNone(scoped_entries(value, "first_598_nt"))


if __name__ == "__main__":
    unittest.main()
