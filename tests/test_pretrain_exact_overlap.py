import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_pretrain_exact_overlap import fasta_sequences, normalize


class PretrainOverlapTest(unittest.TestCase):
    def test_multiline_fasta_and_t_u_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.fa.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(">a\nACGT\nAC\n>b\nUUU\n")
            self.assertEqual(list(fasta_sequences(path)), ["ACGUAC", "UUU"])
        self.assertEqual(normalize("acgtu"), "ACGUU")


if __name__ == "__main__":
    unittest.main()
