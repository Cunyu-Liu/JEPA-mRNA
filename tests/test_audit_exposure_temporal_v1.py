#!/usr/bin/env python3
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import audit_exposure_temporal_v1 as audit


class ExposureTemporalAuditTest(unittest.TestCase):
    def test_scope_guard_rejects_forbidden_inputs(self):
        for name in audit.FORBIDDEN_BASENAMES:
            with self.assertRaises(ValueError):
                audit.guard_scope(Path("/unused") / name)
        for contract in audit.DATASETS.values():
            audit.guard_scope(contract["path"])

    def test_normalization_preserves_ambiguous_symbols(self):
        self.assertEqual(audit.normalize(" acgtun- "), "ACGUUN-")

    def test_fasta_stream_and_length_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.fa.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(">a description\nACG\nTU\n>b\nN-N\n")
            self.assertEqual(list(audit.fasta_records(path)), [("a description", "ACGUU"), ("b", "N-N")])

    def test_exposure_vocabulary_and_label_classes_are_closed(self):
        self.assertEqual(audit.EXPOSURE_STATES, {"EXACT", "HOMOLOGY", "NOT_DETECTED_UNKNOWN", "UNCHECKED"})
        self.assertEqual(audit.LABEL_CLASSES, {"EXPERIMENTAL", "COMPARATIVE_INFERRED", "DATABASE", "THERMODYNAMIC_TEACHER", "SIMULATED"})

    def test_unavailable_view_does_not_fabricate_zero_population(self):
        result = audit.unavailable_view("missing")
        self.assertIsNone(result["n_records"])
        self.assertIsNone(result["state_counts"])
        self.assertEqual(result["state"], "UNCHECKED")

    def test_view_counts_keep_states_separate(self):
        rows = [
            {"views": {"raw": True}, "exposure_state": "EXACT"},
            {"views": {"raw": True}, "exposure_state": "NOT_DETECTED_UNKNOWN"},
            {"views": {"raw": True}, "exposure_state": "HOMOLOGY"},
        ]
        result = audit.exposure_counts(rows, "raw")
        self.assertEqual(result["n_records"], 3)
        self.assertEqual(result["state_counts"]["EXACT"], 1)
        self.assertEqual(result["state_counts"]["NOT_DETECTED_UNKNOWN"], 1)
        self.assertAlmostEqual(result["exact_or_homology_fraction"], 2 / 3)

    def test_view_flags_do_not_use_raw_as_own_or_matched_clean(self):
        matched = {"chanrg_train": {4: False}, "chanrg_validation": {}}
        self.assertEqual(audit.view_flags({"dataset": "chanrg_train", "row_index": 4}, matched), {"raw": True, "own_clean": None, "matched_clean": False})
        self.assertEqual(audit.view_flags({"dataset": "tr1c", "row_index": 0}, matched), {"raw": True, "own_clean": True, "matched_clean": None})

    def test_lineage_never_upgrades_unknown_classes(self):
        rows = [{
            "dataset": "tr1c", "label_class": "DATABASE",
            "raw": {"name": "plk#1", "seq": "ACGU", "source": "ref_plk:plk#1"},
        }]
        result = audit.lineage_metadata(rows)["tr1c"]
        self.assertEqual(result["label_class"], "DATABASE")
        self.assertEqual(result["label_class_counts"]["EXPERIMENTAL"], 0)
        self.assertEqual(result["temporal_state"], "UNCHECKED")
        self.assertEqual(result["modified_parent_state"], "UNCHECKED")
        self.assertEqual(result["unobserved_residue_state"], "UNCHECKED")

    def test_json_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            audit.write_json_new(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                audit.write_json_new(path, {"ok": False})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})


    def test_report_formatter_handles_all_population_states(self):
        summary = {"views": {"tr1c": {
            "raw": {"n_records": 1, "state_counts": {"EXACT": 1, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 0}},
            "matched_clean": {"n_records": None},
        }}}
        lineage = {"datasets": {"tr1c": {"label_class": "DATABASE", "field_coverage": {"source_date": 0, "mmcif_metadata": 0}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            audit.write_report(path, summary, lineage)
            text = path.read_text()
            self.assertIn("| tr1c | raw | 1 | 1 | 0 | 0 | 0 |", text)
            self.assertIn("| tr1c | DATABASE | 0 | 0 | UNCHECKED | UNCHECKED |", text)


if __name__ == "__main__":
    unittest.main()
