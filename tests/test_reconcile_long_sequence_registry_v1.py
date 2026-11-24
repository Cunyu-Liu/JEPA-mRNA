import copy
import json
import unittest
from pathlib import Path

from reconcile_long_sequence_registry_v1 import (
    assert_unrelated_unchanged,
    changed_paths,
    load_audit,
    reconcile_registries,
    row_map,
    validate_reconciliation,
)

REPO = Path("/home/rna_jev_xens2_dualview_round56")
EVIDENCE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/long_sequence_audit_v1")
AUDIT = EVIDENCE / "audit_report.json"
BEFORE_DATA = EVIDENCE / "registry_reconciliation_v1/before_data_registry.json"
BEFORE_BASELINE = EVIDENCE / "registry_reconciliation_v1/before_baseline_registry.json"


class RegistryReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_data = json.loads(BEFORE_DATA.read_text())
        cls.before_baseline = json.loads(BEFORE_BASELINE.read_text())
        cls.audit = load_audit(AUDIT)
        cls.after_data, cls.after_baseline = reconcile_registries(
            cls.before_data, cls.before_baseline, cls.audit)

    def test_only_three_authorized_rows_change(self):
        assert_unrelated_unchanged(self.before_data, self.after_data, "dataset_id",
                                   {"longchain_7224", "longchain_19000"})
        assert_unrelated_unchanged(self.before_baseline, self.after_baseline, "baseline_id",
                                   {"linearpartition"})
        changed_data_ids = {row["dataset_id"] for before, row in zip(
            self.before_data["rows"], self.after_data["rows"]) if before != row}
        changed_baseline_ids = {row["baseline_id"] for before, row in zip(
            self.before_baseline["rows"], self.after_baseline["rows"]) if before != row}
        self.assertEqual(changed_data_ids, {"longchain_7224", "longchain_19000"})
        self.assertEqual(changed_baseline_ids, {"linearpartition"})

    def test_full_file_counts_and_contract_chain_counts_are_distinct(self):
        rows = row_map(self.after_data, "dataset_id")
        self.assertEqual(rows["longchain_7224"]["n_sequences"], 43368)
        self.assertEqual(rows["longchain_7224"]["n_chains_at_contract_length"], 2)
        self.assertEqual(rows["longchain_19000"]["n_sequences"], 202)
        self.assertEqual(rows["longchain_19000"]["n_chains_at_contract_length"], 1)
        self.assertEqual(rows["longchain_7224"]["full_length_coverage"]["chain_lengths"], [7224, 7224])
        self.assertEqual(rows["longchain_19000"]["full_length_coverage"]["chain_lengths"], [19000])

    def test_reconciled_documents_pass_validation(self):
        validate_reconciliation(self.before_data, self.before_baseline,
                                self.after_data, self.after_baseline, self.audit)

    def test_unrelated_row_mutation_is_rejected(self):
        damaged = copy.deepcopy(self.after_data)
        damaged["rows"][0]["description"] = "unrelated mutation"
        with self.assertRaisesRegex(AssertionError, "unrelated registry row changed"):
            validate_reconciliation(self.before_data, self.before_baseline,
                                    damaged, self.after_baseline, self.audit)

    def test_fake_prediction_link_is_rejected(self):
        damaged = copy.deepcopy(self.after_baseline)
        row = row_map(damaged, "baseline_id")["linearpartition"]
        row["prediction_ids"] = ["fabricated/longchain_19000"]
        with self.assertRaises(AssertionError):
            validate_reconciliation(self.before_data, self.before_baseline,
                                    self.after_data, damaged, self.audit)

    def test_fake_prediction_coverage_is_rejected(self):
        damaged = copy.deepcopy(self.after_data)
        row = row_map(damaged, "dataset_id")["longchain_19000"]
        row["full_length_coverage"]["prediction_coverage"] = 1.0
        with self.assertRaises(AssertionError):
            validate_reconciliation(self.before_data, self.before_baseline,
                                    damaged, self.after_baseline, self.audit)

    def test_fake_metric_is_rejected(self):
        damaged = copy.deepcopy(self.after_data)
        row = row_map(damaged, "dataset_id")["longchain_7224"]
        row["metrics"] = {"candidate_pair_recall": 1.0}
        with self.assertRaises(AssertionError):
            validate_reconciliation(self.before_data, self.before_baseline,
                                    damaged, self.after_baseline, self.audit)

    def test_measured_resource_value_is_rejected(self):
        damaged = copy.deepcopy(self.after_baseline)
        row = row_map(damaged, "baseline_id")["linearpartition"]
        row["resource_measurements"]["GEC"]["latency"]["wall_seconds"] = 1.0
        with self.assertRaises(AssertionError):
            validate_reconciliation(self.before_data, self.before_baseline,
                                    self.after_data, damaged, self.audit)

    def test_protocol_and_parameter_variants_are_explicit(self):
        row = row_map(self.after_baseline, "baseline_id")["linearpartition"]
        self.assertEqual(set(row["inference"]["variants"]), {"LinearPartition-C", "LinearPartition-V"})
        self.assertEqual(row["official_weights"]["LinearPartition-C"]["kind"], "EMBEDDED_C_FEATURE_WEIGHTS")
        self.assertEqual(row["official_weights"]["LinearPartition-V"]["kind"], "EMBEDDED_VIENNARNA_PARAMETERS")
        self.assertEqual(row["inference"]["execution_label"], "APPROXIMATE_SPARSE")
        self.assertEqual(row["inference"]["evaluation_status"], "PLANNED_ONLY_NO_PREDICTIONS")

    def test_changed_paths_are_row_scoped(self):
        paths = changed_paths(self.before_data, self.after_data)
        self.assertTrue(paths)
        self.assertTrue(all(path.startswith("rows.") for path in paths))
        self.assertFalse(any(path.startswith("created_utc") or path.startswith("n_rows") for path in paths))


if __name__ == "__main__":
    unittest.main()
