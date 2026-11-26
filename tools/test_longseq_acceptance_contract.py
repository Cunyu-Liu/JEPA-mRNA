import json
import tempfile
import unittest
from pathlib import Path

from longseq_acceptance_contract import (
    classify_execution,
    pair_recall,
    parse_bpp_artifact,
    span_bucket,
    validate_candidate_pairs,
    validate_predictions,
    validate_resources,
)


class AcceptanceContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"device": "cpu", "exact_global_dp": False, "beam_pruning": True,
                         "windowing": False, "top_k_pruning": False, "sparse_decoding": True,
                         "bpp_cutoff": 0.0}
        self.resources = {"latency": {"status": "MEASURED", "wall_seconds": 1.2},
                          "memory": {"status": "MEASURED", "peak_cpu_rss_bytes": 1000,
                                     "peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None},
                          "flops": {"status": "MEASURED", "value": 123, "unit": "FLOPs",
                                    "method": "test counter", "estimated": False}}

    def target(self):
        return {"name": "x", "length": 1200, "sequence_sha256": "seq-hash",
                "pairs": [[0, 49], [0, 50], [10, 209], [20, 1019], [30, 1100]]}

    def record(self, directory):
        path = Path(directory) / "x.bpp"
        rows = [(1, 50, 0.9), (1, 51, 0.8), (11, 210, 0.7), (21, 1020, 0.6), (31, 1101, 0.5)]
        path.write_text(">x\n" + "".join(f"{i} {j} {p}\n" for i, j, p in rows))
        import hashlib
        return {"name": "x", "status": "COMPLETED", "input_length": 1200,
                "input_sequence_sha256": "seq-hash", "output_length": 1200,
                "pairs": [{"pair": [i - 1, j - 1], "probability": p} for i, j, p in rows],
                "pair_artifact": {"path": str(path), "format": "bpp_1based_tsv", "row_count": len(rows),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                "resources": self.resources}

    def test_span_boundaries(self):
        self.assertEqual([span_bucket(0, x) for x in (49, 50, 199, 200, 999, 1000)],
                         ["0-49", "50-199", "50-199", "200-999", "200-999", ">=1000"])

    def test_candidate_bpp_allows_crossing_and_shared_partners(self):
        pairs = [{"pair": [0, 100], "probability": 0.5},
                 {"pair": [1, 101], "probability": 0.4},
                 {"pair": [0, 102], "probability": 0.3}]
        self.assertEqual(len(validate_candidate_pairs(pairs, 103)), 3)
        with self.assertRaises(ValueError):
            validate_candidate_pairs(pairs + pairs[:1], 103)

    def test_exact_and_approximate_columns_cannot_mix(self):
        self.assertEqual(classify_execution(self.protocol), "APPROXIMATE_SPARSE")
        exact = {"device": "cpu", "exact_global_dp": True, "beam_pruning": False,
                 "windowing": False, "top_k_pruning": False, "sparse_decoding": False, "bpp_cutoff": 0.0}
        self.assertEqual(classify_execution(exact), "EXACT_GLOBAL_DP")
        with self.assertRaises(ValueError):
            classify_execution(dict(exact, beam_pruning=True))

    def test_recall_has_separate_long_range_buckets(self):
        reference = [[0, 49], [0, 50], [10, 209], [20, 1019], [30, 1100]]
        result = pair_recall(reference, [{"pair": [0, 50]}, {"pair": [30, 1100]}])
        self.assertEqual(result["overall"]["recall"], 0.4)
        self.assertEqual(result["0-49"]["recall"], 0.0)
        self.assertEqual(result["50-199"]["recall"], 0.5)
        self.assertEqual(result["200-999"]["recall"], 0.0)
        self.assertEqual(result[">=1000"]["recall"], 1.0)

    def test_complete_raw_pairs_pass_full_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.record(directory)
            result = validate_predictions([record], [self.target()], self.protocol)
            self.assertEqual(result["status"], "PASS_FULL_ACCEPTANCE")
            self.assertEqual(result["coverage"]["completed_nucleotide_coverage"], 1.0)
            self.assertEqual(result["micro_pair_recall"]["overall"]["recall"], 1.0)

    def test_summary_truncation_omission_and_pair_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.record(directory)
            for mutation in ({"summary_only": True}, {"output_length": 1199},
                             {"input_sequence_sha256": "wrong"}, {"pairs": []}):
                with self.assertRaises(ValueError):
                    validate_predictions([dict(record, **mutation)], [self.target()], self.protocol)
            with self.assertRaises(ValueError):
                validate_predictions([], [self.target()], self.protocol)

    def test_explicit_failure_is_not_zero_scored_or_complete(self):
        record = {"name": "x", "status": "FAILED_EXPLICIT", "input_length": 1200,
                  "input_sequence_sha256": "seq-hash", "failure_reason": "not run"}
        result = validate_predictions([record], [self.target()], self.protocol)
        self.assertEqual(result["status"], "INCOMPLETE_EXPLICIT_FAILURES")
        self.assertIsNone(result["per_target"][0]["metrics"])
        self.assertEqual(result["coverage"]["record_coverage"], 1.0)
        self.assertEqual(result["coverage"]["completed_target_coverage"], 0.0)

    def test_resources_must_be_measured_not_estimated(self):
        validate_resources(self.resources, "cpu")
        bad = json.loads(json.dumps(self.resources))
        bad["flops"]["estimated"] = True
        with self.assertRaises(ValueError):
            validate_resources(bad, "cpu")
        bad = json.loads(json.dumps(self.resources))
        bad["memory"]["peak_gpu_allocated_bytes"] = 0
        with self.assertRaises(ValueError):
            validate_resources(bad, "cpu")

    def test_bpp_parser_rejects_out_of_full_length_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bpp"
            path.write_text("1 1201 0.5\n")
            with self.assertRaises(ValueError):
                parse_bpp_artifact(path, 1200)


if __name__ == "__main__":
    unittest.main()
