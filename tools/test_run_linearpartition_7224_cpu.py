import tempfile
import unittest
from pathlib import Path

from run_linearpartition_7224_cpu import mea_pairs, parse_gnu_time


class CpuRunTests(unittest.TestCase):
    def test_gnu_time_resources_are_derived_from_raw_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time.txt"
            path.write_text("User time (seconds): 2.50\nSystem time (seconds): 0.25\n"
                            "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.50\n"
                            "Maximum resident set size (kbytes): 2048\nExit status: 0\n")
            result = parse_gnu_time(path)
            self.assertEqual(result["wall_seconds"], 3723.5)
            self.assertEqual(result["peak_cpu_rss_bytes"], 2048 * 1024)
            self.assertEqual(result["user_seconds"], 2.5)

    def test_mea_pairs_cover_nested_structure(self):
        self.assertEqual(mea_pairs("((..))"), [
            {"pair": [0, 5], "probability": None},
            {"pair": [1, 4], "probability": None},
        ])
        with self.assertRaises(ValueError):
            mea_pairs("((..)")


if __name__ == "__main__":
    unittest.main()
