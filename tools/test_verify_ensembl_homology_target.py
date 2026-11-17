import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_ensembl_homology_target as verify


class TargetVerificationTest(unittest.TestCase):
    def test_two_record_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verify.ROOT = root
            verify.EXPECTED_TARGET_COUNT = 2
            output = root / "homology_round56"
            output.mkdir()
            target = output / "target.fa.gz"
            index = output / "index.jsonl.gz"
            rows = []
            with gzip.open(target, "wt") as handle:
                for number, sequence in enumerate(("ACGUACGU", "UUUUGGGG")):
                    handle.write(f">E{number:09d}\n{sequence}\n")
                    rows.append({"target_id": f"E{number:09d}", "length": len(sequence),
                                 "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()})
            with gzip.open(index, "wt") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            verification = root / "ensembl_asset_verification.json"
            verification.write_text("{}")
            manifest = {"status": "PASS_ENSEMBL_HOMOLOGY_TARGET_PREPARED",
                        "records_after_original_length_filter": 2,
                        "verification_sha256": verify.sha256(verification),
                        "target_fasta": str(target), "target_fasta_sha256": verify.sha256(target),
                        "target_index": str(index), "target_index_sha256": verify.sha256(index)}
            (output / "ensembl_target_manifest.json").write_text(json.dumps(manifest))
            verify.main()
            report = json.loads((output / "ensembl_target_qa.json").read_text())
            self.assertEqual(report["n_targets"], 2)


if __name__ == "__main__":
    unittest.main()
