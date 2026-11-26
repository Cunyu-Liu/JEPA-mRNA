#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_module("run_structrfm_gated_full_20261126")
scoring = load_module("score_structrfm_official_bprna_new_20261126")


class GateTest(unittest.TestCase):
    def test_required_free_uses_reserved_peak_and_ten_percent(self):
        self.assertEqual(gate.required_free_mib(1334.0, 32607.0), 5262)
        self.assertEqual(gate.required_free_mib(100.0, 16000.0), 2198)

    def test_probe_and_pass_decision_are_recorded(self):
        probe = json.loads((gate.DEFAULT_ROOT / "reports/longest_489_b1_probe_20261126.json").read_text())
        self.assertEqual(probe["sequence_length"], 489)
        self.assertEqual(probe["batch_size"], 1)
        self.assertEqual(probe["model_forward_device"], "cuda:0")
        self.assertEqual(probe["dp_device"], "cpu")
        decisions = [json.loads(line) for line in (gate.DEFAULT_ROOT / "reports/full_gpu_gate_decisions_20261126.jsonl").read_text().splitlines() if line.strip()]
        self.assertTrue(any(row["eligible"] for row in decisions))
        self.assertFalse(any(row["required_free_mib"] != gate.required_free_mib(row["probe_peak_mib"], row["total_mib"]) for row in decisions))


class IndependentScoreTest(unittest.TestCase):
    def test_micro_macro_and_empty_convention(self):
        labels = [{"name": "a", "seq": "AUGC", "pairs": [[0, 3]]}, {"name": "b", "seq": "AAAA", "pairs": []}]
        predictions = [{"name": "a", "seq": "AUGC", "pred_pairs": [[0, 3], [1, 2]]}, {"name": "b", "seq": "AAAA", "pred_pairs": []}]
        metrics, rows = scoring.score(labels, predictions, require_full=False)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertEqual(metrics["micro_f1"], 2 / 3)
        self.assertEqual(metrics["macro_f1"], (2 / 3 + 1) / 2)
        self.assertEqual(rows[1]["f1"], 1.0)

    def test_partial_coverage_rejected(self):
        with self.assertRaises(ValueError):
            scoring.score([{"name": "a", "seq": "AAAA", "pairs": []}], [])

    def test_real_official_prediction_and_metrics(self):
        root = Path(__file__).parents[1]
        document = json.loads((root / "manifests/foldnar_v2/structrfm_official_bprna_new_results_20261126.json").read_text())
        self.assertTrue(document["full_ready"])
        self.assertEqual(document["metrics"]["coverage"], 1.0)
        self.assertEqual(document["metrics"]["n_with_prediction"], 5388)
        self.assertTrue(document["raw_not_clean"])
        self.assertFalse(document["summary_only_substitution"])
        self.assertEqual(document["historical_prediction"]["status"], "UNCHANGED_BLOCKED")
        rows = [json.loads(line) for line in Path(document["predictions"]["path"]).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 5388)
        self.assertEqual(len({row["name"] for row in rows}), 5388)
        self.assertTrue(all(row["seq"] for row in rows))


if __name__ == "__main__":
    unittest.main()
