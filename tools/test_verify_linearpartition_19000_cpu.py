import tempfile
import unittest
from pathlib import Path

from verify_linearpartition_19000_cpu import parse_time, read_bpp, read_mea, recall


class HoldoutIndependentTests(unittest.TestCase):
    def test_raw_readers_and_span_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            bpp = Path(directory) / "x.bpp"
            bpp.write_text(">x\n1 6 0.5\n")
            self.assertEqual(read_bpp(bpp, 6), [(0, 5, 0.5)])
            mea = Path(directory) / "x.mea"
            mea.write_text(">x\nACGUAU\n((..))\n")
            self.assertEqual(read_mea(mea, "ACGUAU"), ("((..))", [(1, 4), (0, 5)]))
        result = recall([[0, 1000], [0, 50]], [(0, 1000)])
        self.assertEqual(result[">=1000"]["recall"], 1.0)
        self.assertEqual(result["50-199"]["recall"], 0.0)

    def test_time_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time"
            path.write_text("Elapsed (wall clock) time (h:mm:ss or m:ss): 1:39.00\n"
                            "Maximum resident set size (kbytes): 42\n")
            self.assertEqual(parse_time(path), (99.0, 42 * 1024))


if __name__ == "__main__":
    unittest.main()
