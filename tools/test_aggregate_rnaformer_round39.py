import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import aggregate_rnaformer_round39 as aggregate


class EmptyCleanSplitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "ref_pdb_ts3.jsonl"
        self.data.write_bytes(b"")
        self.audit = self.root / "audit.json"
        self.audit.write_text(json.dumps({
            "status": "PASS",
            "split_sha256": aggregate.EXPECTED_SPLIT_SHA256["ref_pdb_ts3"],
            "clean_sha256": hashlib.sha256(b"").hexdigest(),
            "clean_n": 0,
            "raw_n": aggregate.EXPECTED_SPLIT_COUNTS["ref_pdb_ts3"],
            "removed_exact": aggregate.EXPECTED_SPLIT_COUNTS["ref_pdb_ts3"],
            "removed_homology_only": 0,
        }))
        self.result = {
            "schema_version": 1,
            "status": "EMPTY_AFTER_DELEAK",
            "model": "rnaformer",
            "protocol": {
                "data": str(self.data),
                "data_sha256": hashlib.sha256(b"").hexdigest(),
                "clean_audit": str(self.audit),
                "n_sequences": 0,
            },
            "f1_runs": [],
            "runs": [],
            "predictions": [],
        }
        self.path = self.root / "result.json"
        self.write(self.result)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, result):
        self.path.write_text(json.dumps(result))

    def validate(self):
        return aggregate.validate(self.path, "ref_pdb_ts3", True, {})

    def test_empty_clean_split_has_no_inference_protocol(self):
        output = self.validate()
        self.assertEqual(output["status"], "EMPTY_AFTER_DELEAK")
        self.assertEqual(output["n"], 0)
        self.assertNotIn("strict_micro_f1", output)
        self.assertEqual(output["source_result"], str(self.path))
        self.assertEqual(output["source_data"], str(self.data))
        self.assertEqual(output["result_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_empty_split_rejects_inconsistent_clean_n(self):
        result = copy.deepcopy(self.result)
        result["protocol"]["n_sequences"] = 1
        self.write(result)
        with self.assertRaises(AssertionError):
            self.validate()

    def test_empty_split_rejects_predictions(self):
        result = copy.deepcopy(self.result)
        result["predictions"] = [{"name": "unexpected", "pairs": []}]
        self.write(result)
        with self.assertRaises(AssertionError):
            self.validate()

    def test_empty_raw_split_is_rejected(self):
        with self.assertRaises(AssertionError):
            aggregate.validate(self.path, "ref_pdb_ts3", False, {})

    def test_nonempty_inference_still_requires_cuda(self):
        self.data.write_text(json.dumps({"name": "x", "pairs": []}) + "\n")
        result = copy.deepcopy(self.result)
        result["protocol"].update({
            "data_sha256": hashlib.sha256(self.data.read_bytes()).hexdigest(),
            "n_sequences": 1,
            "cached_embeddings": False,
        })
        self.write(result)
        with self.assertRaises(KeyError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
