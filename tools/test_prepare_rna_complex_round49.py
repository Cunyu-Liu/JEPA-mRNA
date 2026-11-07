import unittest

from prepare_rna_complex_round49 import nested_canonical, one_letter, parse_nt


class RnaComplexTests(unittest.TestCase):
    def test_dssr_modified_and_negative_insertion_ids(self):
        self.assertEqual(parse_nt("A.5MC40"), ("A", 40, "", "5MC"))
        self.assertEqual(parse_nt("B.2MG-3A"), ("B", -3, "A", "2MG"))
        with self.assertRaises(ValueError):
            parse_nt("A.C?")

    def test_nested_and_shared_pair_rejections(self):
        self.assertEqual(nested_canonical([[1, 8], [2, 7]]), [[1, 8], [2, 7]])
        self.assertEqual(nested_canonical([[1, 8], [2, 9]]), [])
        self.assertEqual(nested_canonical([[1, 8], [1, 9]]), [])

    def test_modified_parent_requires_explicit_evidence(self):
        self.assertEqual(one_letter("PSU", {}), "U")
        self.assertEqual(one_letter("5MC", {"5MC": {"parent": "C"}}), "C")
        with self.assertRaises(ValueError):
            one_letter("5MC", {})
        with self.assertRaises(ValueError):
            one_letter("UNKNOWN", {})


if __name__ == "__main__":
    unittest.main()
