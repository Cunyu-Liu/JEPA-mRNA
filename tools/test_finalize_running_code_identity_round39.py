import json
import tempfile
import unittest
from pathlib import Path

from finalize_running_code_identity_round39 import diff_paths, finalize


class IdentityCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.result = root / "result.json"
        self.pin = root / "pin.json"
        self.backup = root / "backup.json"
        self.audit = root / "audit.json"
        self.payload = {"status": "PASS", "protocol": {"code_commit": "new", "code_sha256": "new-sha",
                                                       "device_type": "cuda"},
                        "runs": [{"repeat": i, "strict_micro_f1": 0.5} for i in range(3)],
                        "predictions": [{"name": "a", "pairs": [[0, 3]]}]}
        self.result.write_text(json.dumps(self.payload, indent=1) + "\n")
        self.pin.write_text(json.dumps({
            "status": "RUNNING_PROCESS_CODE_IDENTITY_PINNED", "code_commit": "old", "code_sha256": "old-sha",
            "correction_scope": "protocol.code_sha256 and protocol.code_commit only; predictions/runs/timing immutable",
            "process_command": f"python speed.py --out {self.result}"}))

    def tearDown(self):
        self.directory.cleanup()

    def test_only_identity_changes_and_idempotent(self):
        original_bytes = self.result.read_bytes()
        first = finalize(self.result, self.pin, self.backup, self.audit)
        first_hash = first["result_sha256"]
        self.assertEqual(self.backup.read_bytes(), original_bytes)
        corrected = json.loads(self.result.read_text())
        self.assertEqual(corrected["protocol"]["code_commit"], "old")
        self.assertEqual(corrected["protocol"]["code_sha256"], "old-sha")
        self.assertEqual(corrected["runs"], self.payload["runs"])
        self.assertEqual(corrected["predictions"], self.payload["predictions"])
        second = finalize(self.result, self.pin, self.backup, self.audit)
        self.assertEqual(second["result_sha256"], first_hash)
        self.assertEqual(second["changed_paths"], ["protocol.code_commit", "protocol.code_sha256"])

    def test_wrong_result_path_fails(self):
        other = self.result.parent / "other.json"
        other.write_text(self.result.read_text())
        with self.assertRaises(AssertionError):
            finalize(other, self.pin, self.backup, self.audit)

    def test_diff_detects_nested_change(self):
        self.assertEqual(diff_paths({"a": [{"b": 1}]}, {"a": [{"b": 2}]}), {"a[0].b"})


if __name__ == "__main__":
    unittest.main()
