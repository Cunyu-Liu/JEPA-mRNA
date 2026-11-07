import itertools
import random
import unittest

from prepare_rna_complex_round49 import nested_canonical, observed_residue_map, one_letter, parse_nt


class RnaComplexTests(unittest.TestCase):
    def test_dssr_modified_and_negative_insertion_ids(self):
        self.assertEqual(parse_nt("A.5MC40"), ("A", 40, "", "5MC"))
        self.assertEqual(parse_nt("B.2MG-3A"), ("B", -3, "A", "2MG"))
        with self.assertRaises(ValueError):
            parse_nt("A.C?")

    def test_nested_and_shared_pair_rejections(self):
        self.assertEqual(nested_canonical([[1, 8], [2, 7]]), [[1, 8], [2, 7]])
        self.assertEqual(nested_canonical([[1, 8], [2, 9]]), [[1, 8]])
        self.assertEqual(nested_canonical([[1, 8], [1, 9]]), [[1, 8]])
        self.assertEqual(nested_canonical([[1, 4], [2, 8], [5, 7]]), [[1, 4], [5, 7]])

    def test_nested_canonical_matches_small_brute_force(self):
        random.seed(19)
        candidates = list(itertools.combinations(range(1, 8), 2))
        for _ in range(12):
            pairs = random.sample(candidates, 9)
            result = nested_canonical(pairs)
            optimum = 0
            for mask in range(1 << len(pairs)):
                chosen = [pairs[index] for index in range(len(pairs)) if mask & (1 << index)]
                endpoints = [position for pair in chosen for position in pair]
                crossing = any(a < c < b < d for a, b in chosen for c, d in chosen)
                if not crossing and len(endpoints) == len(set(endpoints)):
                    optimum = max(optimum, len(chosen))
            self.assertEqual(len(result), optimum)

    def test_modified_parent_requires_explicit_evidence(self):
        self.assertEqual(one_letter("PSU", {}), "U")
        self.assertEqual(one_letter("5MC", {"5MC": {"parent": "C"}}), "C")
        with self.assertRaises(ValueError):
            one_letter("5MC", {})
        with self.assertRaises(ValueError):
            one_letter("UNKNOWN", {})

    def test_observed_map_does_not_trust_author_numbering(self):
        mapping = {"A": {1475: {"seq_id": 1475, "letter": "A", "observed": True},
                         1476: {"seq_id": 1476, "letter": "U", "observed": True}}}
        dssr = {"nts": [{"index": 1, "nt_type": "RNA", "nt_id": "A.A1475", "nt_name": "A"},
                        {"index": 2, "nt_type": "RNA", "nt_id": "A.U1476", "nt_name": "U"}]}
        result = observed_residue_map(dssr, mapping, {})
        self.assertEqual(result["A.A1475"]["seq_id"], 1475)
        with self.assertRaises(ValueError):
            observed_residue_map({"nts": dssr["nts"][:1]}, mapping, {})


if __name__ == "__main__":
    unittest.main()
