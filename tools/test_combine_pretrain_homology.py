import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import combine_pretrain_homology as combine


class CombineHomologyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        combine.OUTPUT = self.root
        (self.root / "query_manifest.json").write_text(json.dumps({
            "n_unique_queries": 2,
            "queries": {"setA": {"homology_queries": 1}, "setB": {"homology_queries": 1}},
        }))
        for source in combine.SOURCES:
            self.write_source(source)

    def tearDown(self):
        self.directory.cleanup()

    def write_source(self, source, second_sets=None):
        directory = self.root / source
        directory.mkdir(exist_ok=True)
        per_query = directory / "query_homology.jsonl.gz"
        rows = [{"query_id": "Q1", "source_sets": ["setA"], "detected": True,
                 "best_alignment": {"target": "T1"}},
                {"query_id": "Q2", "source_sets": second_sets or ["setB"], "detected": False,
                 "best_alignment": None}]
        with gzip.open(per_query, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        summary = {"status": "PASS_HOMOLOGY_SEARCH_AGGREGATED_NOT_FAMILY_INDEPENDENCE",
                   "per_query_path": str(per_query), "per_query_sha256": combine.sha256(per_query),
                   "detected_homology_unique": 1, "queries": {}}
        (directory / "homology_summary.json").write_text(json.dumps(summary))

    def test_union_preserves_unknown(self):
        combine.main()
        report = json.loads((self.root / "homology_union_summary.json").read_text())
        self.assertEqual(report["detected_by_any_source"], 1)
        self.assertEqual(report["not_detected_by_any_source"], 1)
        with gzip.open(self.root / "query_homology_union.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(rows[1]["interpretation"], "NOT_DETECTED_LIMITED_SOURCES_NT_HISTORY_UNRESOLVED")

    def test_membership_mismatch_rejected(self):
        self.write_source(combine.SOURCES[-1], second_sets=["different"])
        with self.assertRaises(AssertionError):
            combine.main()


if __name__ == "__main__":
    unittest.main()
