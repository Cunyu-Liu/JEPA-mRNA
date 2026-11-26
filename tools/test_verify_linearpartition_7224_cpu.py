import tempfile
import unittest
from pathlib import Path

from verify_linearpartition_7224_cpu import parse_time, read_bpp, read_mea, recall


class IndependentVerificationTests(unittest.TestCase):
    def test_independent_bpp_and_mea_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            bpp = Path(directory) / "x.bpp"
            bpp.write_text(">x\n1 6 0.5\n2 5 0.25\n")
            self.assertEqual(read_bpp(bpp, 6), [(0, 5, 0.5), (1, 4, 0.25)])
            mea = Path(directory) / "x.mea"
            mea.write_text(">x\nACGUAU\n((..))\n")
            self.assertEqual(read_mea(mea, "ACGUAU"), ("((..))", [(1, 4), (0, 5)]))

    def test_independent_span_recall(self):
        result = recall([[0, 49], [0, 50], [0, 1000]], [(0, 50)])
        self.assertEqual(result["overall"]["recall"], 1 / 3)
        self.assertEqual(result["0-49"]["recall"], 0.0)
        self.assertEqual(result["50-199"]["recall"], 1.0)
        self.assertEqual(result[">=1000"]["recall"], 0.0)
        self.assertIsNone(result["200-999"]["recall"])

    def test_independent_time_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time"
            path.write_text("Elapsed (wall clock) time (h:mm:ss or m:ss): 2:03.25\n"
                            "Maximum resident set size (kbytes): 17\n")
            self.assertEqual(parse_time(path), (123.25, 17 * 1024))


if __name__ == "__main__":
    unittest.main()
