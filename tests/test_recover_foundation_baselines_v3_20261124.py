#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "recover_foundation_baselines_v3_20261124.py"
SPEC = importlib.util.spec_from_file_location("foundation_recovery_v3", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExactPairScoringTest(unittest.TestCase):
    def test_micro_macro_and_full_coverage(self):
        rows = [
            {"name": "a", "seq": "AAAA", "pairs": [[0, 3]]},
            {"name": "b", "seq": "AAAA", "pairs": []},
        ]
        predictions = {"a": {"pairs": [[0, 3], [1, 2]]}, "b": {"pairs": []}}
        metrics, per_sequence = MODULE.strict_metrics(rows, predictions)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertEqual(metrics["micro_precision"], 0.5)
        self.assertEqual(metrics["micro_recall"], 1.0)
        self.assertEqual(metrics["micro_f1"], 2 / 3)
        self.assertEqual(metrics["macro_f1"], (2 / 3 + 1) / 2)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(per_sequence[1]["f1"], 1.0)

    def test_missing_prediction_is_rejected_not_imputed(self):
        rows = [{"name": "a", "seq": "AAAA", "pairs": []}]
        with self.assertRaises(AssertionError):
            MODULE.strict_metrics(rows, {})

    def test_extra_prediction_is_rejected(self):
        rows = [{"name": "a", "seq": "AAAA", "pairs": []}]
        with self.assertRaises(AssertionError):
            MODULE.strict_metrics(rows, {"a": {"pairs": []}, "b": {"pairs": []}})

    def test_partner_conflict_is_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 3], [0, 2]], 4)

    def test_out_of_bounds_pair_is_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.pair_set([[0, 4]], 4)


class StructRFMConversionTest(unittest.TestCase):
    def test_one_based_connects_and_order_bound_names(self):
        rows = [{"name": "x", "seq": "AUGC", "pairs": [[0, 3]]}]
        document = {
            "n": 1,
            "seqs": [{"id": 0, "seq": "AUGC", "connects": [4, 0, 0, 1]}],
            "per_seq": [{"tp": 1, "fp": 0, "fn": 0}],
        }
        predictions = MODULE.parse_structrfm(document, rows)
        self.assertEqual(predictions, [{"name": "x", "pairs": [[0, 3]]}])

    def test_sequence_mismatch_is_rejected(self):
        rows = [{"name": "x", "seq": "AAAA", "pairs": []}]
        document = {"n": 1, "seqs": [{"id": 0, "seq": "UUUU", "connects": [0] * 4}], "per_seq": [{}]}
        with self.assertRaises(AssertionError):
            MODULE.parse_structrfm(document, rows)


class ProtocolIdentityTest(unittest.TestCase):
    def test_author_and_project_nucleicbert_arms_are_distinct(self):
        author = MODULE.SOURCES["nucleicbert_author_secstr_full_ft"]
        project = MODULE.SOURCES["nucleicbert_project_frozen_encoder_tr1c"]
        self.assertIn("frozen=False", author["training_protocol"])
        self.assertIn("encoder frozen", project["training_protocol"])
        self.assertNotEqual(author["sha256"], project["sha256"])

    def test_rnafm_arm_is_not_mislabeled_official_resnet(self):
        spec = MODULE.SOURCES["rnafm_project_frozen_encoder_crf"]
        self.assertEqual(spec["origin"], "project_downstream_checkpoint")
        self.assertIn("not the official RNA-FM-ResNet", spec["comparability"])


if __name__ == "__main__":
    unittest.main()
