import unittest

from verify_lineardesign_candidate_round49 import evaluate

TABLE = "/home/rna_jev/third_party/LinearDesign/codon_usage_freq_table_human.csv"


class LinearDesignEvaluationTests(unittest.TestCase):
    def test_official_lambda0_example(self):
        result = evaluate("AUGAACGAUACGGAGGCGAUC", "MNDTEAI", TABLE)
        self.assertEqual(result["translated"], "MNDTEAI")
        self.assertEqual(result["structure"], "......(((.((....)))))")
        self.assertAlmostEqual(result["mfe_kcal_mol"], -1.40, places=2)
        self.assertAlmostEqual(result["cai"], 0.695, places=3)
        self.assertEqual(result["rna_version"], "2.7.2")

    def test_translation_and_internal_stop_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluate("AUGAACGAUACGGAGGCGAUC", "MNDTEAA", TABLE)
        with self.assertRaises(ValueError):
            evaluate("AUGUAAAUG", "M*M", TABLE)


if __name__ == "__main__":
    unittest.main()
