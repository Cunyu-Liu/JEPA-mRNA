import tempfile
import unittest
from pathlib import Path

from recover_linearpartition_longseq import (
    classify_execution,
    full_length_structures,
    official_short_sequence,
    parse_bpp,
    sha256_text,
)


class RecoveryTests(unittest.TestCase):
    def test_official_sequence_identity(self):
        sequence = "GCGGGAAUAGCUCAGUUGGUAGAGCACGACCUUGCCAAGGUCGGGGUCGCGAGUUCGAGUCUCGUUUCCCGCUCCA"
        self.assertEqual(len(sequence), 76)
        self.assertEqual(sha256_text(sequence), "2af42dfab28438539d9cfeb851d53541bc11fdbc2ee87e1b5d97db4f5d1bdf5d")

    def test_bpp_parser_preserves_complete_emitted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.bpp"
            path.write_text(">official\n1 4 0.99\n2 3 2.5e-1\n")
            self.assertEqual(parse_bpp(path, 4), [
                {"pair_1based_inclusive": [1, 4], "probability": 0.99},
                {"pair_1based_inclusive": [2, 3], "probability": 0.25},
            ])
            for content in ("1 5 0.5\n", "1 4 nan\n", "1 4 0.5\n1 4 0.6\n", "1 4\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    parse_bpp(path, 4)

    def test_structure_must_be_balanced_and_full_length(self):
        self.assertEqual(full_length_structures("x\n((..))\n", 6), ["((..))"])
        for stdout in ("((..)\n", ")(...\n", "((...)\n", "no structure\n"):
            with self.assertRaises(ValueError):
                full_length_structures(stdout, 6)

    def test_default_protocol_is_not_exact(self):
        self.assertEqual(classify_execution(100, 0.0)["column"], "APPROXIMATE_SPARSE")
        with self.assertRaises(ValueError):
            classify_execution(0, 0.0)


if __name__ == "__main__":
    unittest.main()
