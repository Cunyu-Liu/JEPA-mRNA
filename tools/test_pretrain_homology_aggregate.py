import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from aggregate_pretrain_homology import summarize_matches


class HomologyAggregateTest(unittest.TestCase):
    def setUp(self):
        self.queries = {"Q1": {"query_id": "Q1", "length": 100}, "Q2": {"query_id": "Q2", "length": 50}}

    def line(self, query="Q1", target="T1", pident="80.000", alnlen="80", qlen="100", tlen="100"):
        return f"{query}\t{target}\t{pident}\t{alnlen}\t{qlen}\t{tlen}\t1\t80\t1\t80\t1e-20\t150\n"

    def test_threshold_and_deterministic_best(self):
        best, count = summarize_matches([self.line(), self.line(target="T2", pident="99.000")], self.queries)
        self.assertEqual(count, 2)
        self.assertEqual(best["Q1"]["target"], "T2")

    def test_tied_best_uses_target_id(self):
        best, _ = summarize_matches([self.line(target="T2"), self.line(target="T1")], self.queries)
        self.assertEqual(best["Q1"]["target"], "T1")

    def test_no_hit_is_not_homology_exclusion(self):
        best, count = summarize_matches([], self.queries)
        self.assertEqual((best, count), ({}, 0))

    def test_reject_below_threshold_and_unknown_identity(self):
        with self.assertRaises(AssertionError):
            list(summarize_matches([self.line(pident="79.999")], self.queries))
        with self.assertRaises(AssertionError):
            list(summarize_matches([self.line(query="Q3")], self.queries))
        with self.assertRaises(AssertionError):
            list(summarize_matches(["malformed\n"], self.queries))


if __name__ == "__main__":
    unittest.main()
