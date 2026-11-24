import json
import tempfile
import unittest
from pathlib import Path

from verify_long_sequence_linearpartition_audit_v1 import (
    approximation_label,
    candidate_and_helix_recall,
    normalise_sequence,
    parse_linearpartition_bpp,
    reference_helices,
    source_to_frozen_row,
    validate_full_coverage,
    validate_zero_based_pairs,
)


class LongSequenceAuditUnitTests(unittest.TestCase):
    def test_sequence_normalisation_preserves_every_position(self):
        self.assertEqual(normalise_sequence("aTgcnU"), "AUGCNU")

    def test_source_conversion_changes_coordinates_not_raw_labels(self):
        source = {
            "name": "chain", "seq": "ATGCU", "pairs": [[1, 5], [2, 4]],
            "canonical_nested_pairs": [[1, 5], [2, 4]], "raw_pairs": [[1, 5], [2, 3]],
            "source": {"canonical_nested_pairs": [[1, 5], [2, 4]]},
        }
        expected = dict(source)
        expected.update({"seq": "AUGCU", "pairs": [[0, 4], [1, 3]],
                         "canonical_nested_pairs": [[0, 4], [1, 3]]})
        self.assertEqual(source_to_frozen_row(source), expected)
        self.assertEqual(source["pairs"], [[1, 5], [2, 4]])

    def test_pair_validation_rejects_truncation_conflict_and_crossing(self):
        validate_zero_based_pairs([(0, 9), (1, 8)], 10)
        for pairs in ([(0, 10)], [(0, 9), (9, 8)], [(0, 5), (5, 9)], [(0, 5), (1, 6)]):
            with self.assertRaises(ValueError):
                validate_zero_based_pairs(pairs, 10)

    def test_reference_helices_are_maximal_stacks(self):
        helices, isolated = reference_helices([(0, 9), (1, 8), (2, 7), (4, 5)])
        self.assertEqual(len(helices), 1)
        self.assertEqual(helices[0]["length"], 3)
        self.assertEqual(helices[0]["pairs"], [[0, 9], [1, 8], [2, 7]])
        self.assertEqual(isolated, 1)

    def test_candidate_and_complete_helix_recall(self):
        reference = [(0, 9), (1, 8), (2, 7), (4, 5)]
        result = candidate_and_helix_recall(reference, [(0, 9), (1, 8), (4, 5)])
        self.assertEqual(result["candidate_pair_recall"], 0.75)
        self.assertEqual(result["complete_helix_recall"], 0.0)
        result = candidate_and_helix_recall(reference, reference)
        self.assertEqual(result["complete_helix_recall"], 1.0)
        self.assertEqual(result["isolated_reference_pair_count_excluded_from_helix_metric"], 1)

    def test_empty_reference_recall_is_unavailable_not_zero(self):
        result = candidate_and_helix_recall([], [])
        self.assertIsNone(result["candidate_pair_recall"])
        self.assertIsNone(result["complete_helix_recall"])

    def test_linearpartition_bpp_is_converted_from_one_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.bpp"
            path.write_text("1 4 9.9e-01\n2 3 0.25\n")
            pairs = parse_linearpartition_bpp(path, 4)
            self.assertEqual(pairs, [{"pair": [0, 3], "probability": 0.99},
                                     {"pair": [1, 2], "probability": 0.25}])
            path.write_text("1 5 0.5\n")
            with self.assertRaises(ValueError):
                parse_linearpartition_bpp(path, 4)

    def test_full_coverage_requires_all_identities_and_lengths(self):
        expected = {"a": {"length": 4, "sequence_sha256": "hash-a"},
                    "b": {"length": 5, "sequence_sha256": "hash-b"}}
        records = [
            {"name": "a", "input_length": 4, "input_sequence_sha256": "hash-a",
             "status": "COMPLETED", "output_length": 4, "pairs": [[0, 3]]},
            {"name": "b", "input_length": 5, "input_sequence_sha256": "hash-b",
             "status": "FAILED_EXPLICIT"},
        ]
        self.assertEqual(validate_full_coverage(records, expected)["n_targets"], 2)
        with self.assertRaises(ValueError):
            validate_full_coverage(records[:1], expected)
        truncated = [dict(records[0], output_length=3), records[1]]
        with self.assertRaises(ValueError):
            validate_full_coverage(truncated, expected)

    def test_approximation_label_cannot_be_claimed_for_sparse_execution(self):
        self.assertEqual(approximation_label(True, False), "EXACT_GLOBAL_DP")
        self.assertEqual(approximation_label(True, True), "APPROXIMATE_SPARSE")
        self.assertEqual(approximation_label(False, False), "APPROXIMATE_SPARSE")


if __name__ == "__main__":
    unittest.main()
