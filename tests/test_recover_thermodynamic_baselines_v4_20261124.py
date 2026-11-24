#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "recover_thermodynamic_baselines_v4_20261124.py"
SPEC = importlib.util.spec_from_file_location("thermodynamic_recovery_v4", MODULE_PATH)
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


class DecoderParserTest(unittest.TestCase):
    def test_dot_bracket_pairs(self):
        self.assertEqual(MODULE.parse_dot_pairs("(..(....))"), [(0, 9), (3, 8)])

    def test_unbalanced_dot_bracket_is_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.parse_dot_pairs("(....")

    def test_only_first_ct_block_is_decoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "multi.ct"
            block = "3 energy\n1 A 0 2 3 1\n2 U 1 3 0 2\n3 G 2 0 1 3\n"
            path.write_text(block + block)
            pairs, blocks = MODULE.parse_ct_pairs(path, "AUG")
            self.assertEqual(pairs, [(0, 2)])
            self.assertEqual(blocks, 2)


class ScopeAndProtocolTest(unittest.TestCase):
    def test_six_decoder_identities_are_not_merged(self):
        self.assertEqual(len(MODULE.SYSTEMS), 6)
        self.assertEqual(len(set(MODULE.SYSTEMS)), 6)
        self.assertIn("rnastructure_fold", MODULE.SYSTEMS)
        self.assertIn("rnastructure_maxexpect", MODULE.SYSTEMS)
        self.assertNotIn("rnastructure", MODULE.SYSTEMS)

    def test_unavailable_clean_assets_are_null(self):
        clean = MODULE.unavailable_clean("not applicable")
        for key in ("scope", "source_artifact", "source_artifact_sha256", "prediction_artifact", "prediction_artifact_sha256", "metrics", "coverage"):
            self.assertIsNone(clean[key])

    def test_clean_contract_is_distinct_from_raw_5388(self):
        self.assertNotEqual(MODULE.DATA, MODULE.CLEAN_DATA)
        self.assertNotEqual(MODULE.DATA_SHA256, MODULE.CLEAN_DATA_SHA256)
        self.assertNotEqual(MODULE.RAW_SOURCES["contrafold"][1], MODULE.CLEAN_CONTRAFOLD[1])

    def test_replay_is_bounded_and_cpu_only(self):
        self.assertEqual(len(MODULE.REPLAY_INDICES), 12)
        self.assertEqual(MODULE.REPLAY_INDICES[-1], 5387)

    def test_miniature_recorded_artifact_audit(self):
        rows = [{"name": "a", "seq": "AAAA", "pairs": [[0, 3]]}]
        identity = [["a", [[0, 3]]]]
        prediction_sha = hashlib.sha256(MODULE.canonical(identity).encode()).hexdigest()
        per_sequence = [{"name": "a", "tp": 1, "fp": 0, "fn": 0, "f1": 1.0, "pred_pairs": [[0, 3]]}]
        repeats = [{"prediction_sha256": prediction_sha, "metrics": {"n_sequences": 1, "tp": 1, "fp": 0, "fn": 0, "micro_f1": 1.0, "macro_f1": 1.0}} for _ in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps({"per_sequence": per_sequence}))
            result = MODULE.audit_frozen("toy", "toy_decode", "raw", source, MODULE.sha256(source), rows, per_sequence, repeats, root)
            self.assertEqual(result["metrics"]["micro_f1"], 1.0)
            self.assertEqual(result["recorded_deterministic_repeats_verified"], 3)
            self.assertEqual(result["prediction_identity_sha256"], prediction_sha)


if __name__ == "__main__":
    unittest.main()
