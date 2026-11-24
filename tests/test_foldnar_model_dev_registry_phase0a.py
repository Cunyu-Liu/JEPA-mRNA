import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_foldnar_model_dev_registry_phase0a import (
    grouped_holdout, leakage, pair_features, stable_id, summarize_records,
)


class FoldNarModelDevRegistryPhase0ATest(unittest.TestCase):
    def test_stable_ids_distinguish_duplicate_names(self):
        first = stable_id("bprna_tr1c", 4, "same.bpseq")
        second = stable_id("bprna_tr1c", 9, "same.bpseq")
        self.assertNotEqual(first, second)
        self.assertEqual(first, "bprna_tr1c:row000004:same.bpseq")

    def test_group_holdout_is_deterministic_and_group_disjoint(self):
        groups = [index // 2 for index in range(20)]
        first, dev = grouped_holdout(groups, "unit")
        second, second_dev = grouped_holdout(groups, "unit")
        self.assertEqual(first, second)
        self.assertEqual(dev, second_dev)
        self.assertEqual(len(dev), 2)
        self.assertTrue(all((group in dev) == (assignment == "dev") for group, assignment in zip(groups, first)))

    def test_nested_hairpin_and_stacked_pairs(self):
        features = pair_features(6, [(0, 5), (1, 4)])
        self.assertEqual(features["n_pairs"], 2)
        self.assertEqual(features["isolated_pair_counts"], {"stacked": 2})
        self.assertEqual(features["loop_closure_counts"], {"hairpin": 1, "internal_or_bulge": 1})
        self.assertEqual(features["topology"], "single_root_no_multiloop")

    def test_isolated_pairs_and_multi_root_topology(self):
        features = pair_features(7, [(0, 2), (4, 6)])
        self.assertEqual(features["isolated_pair_counts"], {"isolated": 2})
        self.assertEqual(features["pair_span_counts"], {"le3": 2})
        self.assertEqual(features["topology"], "multi_root_no_multiloop")

    def test_multiloop_closing_and_member_pairs(self):
        features = pair_features(12, [(0, 11), (1, 5), (6, 10)])
        self.assertEqual(features["loop_closure_counts"], {"hairpin": 2, "multiloop": 1})
        self.assertEqual(features["junction_pair_counts"], {"closing_order_2": 1, "member_order_2": 2})
        self.assertEqual(features["topology"], "contains_multiloop")

    def test_partner_conflicts_and_invalid_coordinates_fail(self):
        with self.assertRaises(ValueError):
            pair_features(8, [(0, 7), (0, 6)])
        with self.assertRaises(ValueError):
            pair_features(8, [(-1, 7)])

    def test_small_concentrated_environment_is_diagnostic_only(self):
        pair_counts = {"4_7": 20}
        record = {
            "source": "bpseq:x", "family_id": "RF00001", "family_status": "PASS",
            "cluster_id": 7, "length": 100, "topology": "single_root_no_multiloop",
            "n_pairs": 20, "pair_span_counts": pair_counts,
            "isolated_pair_counts": {"stacked": 20}, "loop_closure_counts": {"hairpin": 20},
            "junction_pair_counts": {},
        }
        summary = summarize_records([record])
        self.assertEqual(summary["n_sequences"], 1)
        self.assertEqual(summary["n_pairs"], 20)
        self.assertTrue(all(not row["route_eligible"] for row in summary["environments"]))
        span = next(row for row in summary["environments"] if row["environment"] == "pair_span")
        self.assertEqual(span["n_pairs"], 20)
        self.assertEqual(span["top_known_family_fraction_of_all"], 1.0)
        self.assertIn("min_sequences", span["route_failure_reasons"])

    def test_leakage_states_do_not_imply_clean(self):
        state = leakage("NOT_DETECTED_UNKNOWN", "NOT_DETECTED_UNKNOWN")
        allowed = {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"}
        self.assertTrue(all(value in allowed for key, value in state.items() if key != "interpretation"))
        self.assertEqual(state["pretraining_exposure"], "UNCHECKED")
        self.assertIn("not a clean claim", state["interpretation"])


if __name__ == "__main__":
    unittest.main()
