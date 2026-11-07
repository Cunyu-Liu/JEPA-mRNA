import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare_ripe_paired_round48 import SPLITS


class ComplexDecisionIntegrationTests(unittest.TestCase):
    def test_full_pipeline_with_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prep = root / "data/rna_complex_round49/d1_preparation"
            prep.mkdir(parents=True)
            config = {"d1": {"data": "train_d1", "teacher_dir": "teacher"},
                      "complex_holdout_sha256": "holdout"}
            (prep / "d1_config.json").write_text(json.dumps(config))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {"selected": {"experiment": "d0exp", "run_prefix": "d0_s", "mean_best_vl0": 20.0},
                        "arms": [{"arm": "complex_d1"}], "d1_config_sha256": digest(prep / "d1_config.json")}
            (prep / "comparison_manifest.json").write_text(json.dumps(manifest))
            for seed in range(3):
                run = root / f"runs/complex_d1_s{seed}"
                run.mkdir(parents=True)
                meta = {"converged": True, "early_stopped": True, "status": "completed",
                        "schedule": {"steps": 0}, "validation": {"patience": 6},
                        "data": "train_d1", "objective": {"teacher": "teacher"},
                        "scorer": "pairmlp", "best_validation_loss": 18.0 + seed}
                (run / "run_meta.json").write_text(json.dumps(meta))
                for experiment in ["d0exp", "complex_d1"]:
                    for split in SPLITS + ["complex_holdout"]:
                        output_experiment = "complex_d0" if experiment == "d0exp" and split == "complex_holdout" else experiment
                        directory = root / "eval_decision" / output_experiment / f"s{seed}"
                        directory.mkdir(parents=True, exist_ok=True)
                        improved = experiment == "complex_d1" and split in set(SPLITS[8:])
                        row = {"name": "synthetic", "length": 100, "tp": 7 if improved else 5,
                               "fp": 2, "fn": 1 if improved else 3,
                               "span_30": {"tp": 2, "fp": 1, "fn": 1},
                               "span_100": {"tp": 1, "fp": 1, "fn": 0}}
                        result = {"status": "PASS_SYNTHETIC", "data_sha256": "holdout" if split == "complex_holdout" else split,
                                  "raw_n": 30, "n": 30, "n_long_unsupported": 0,
                                  "checkpoint_sha256": "checkpoint",
                                  "per_sequence": [dict(row, name=f"synthetic_{i}") for i in range(30)]}
                        (directory / f"{split}.json").write_text(json.dumps(result))
                for experiment in ["d0exp", "complex_d1"]:
                    directory = root / "eval_decision" / experiment
                    (directory / "summary.json").write_text(json.dumps({"status": "PASS_SYNTHETIC"}))
                    (directory / "confidence_summary.json").write_text(json.dumps({"status": "PASS_SYNTHETIC"}))
                    (directory / "sota_comparison.json").write_text(json.dumps({"status": "PASS_SYNTHETIC", "main_sota_any_wins": 0}))
            for decoder in ["greedy", "turner0"]:
                directory = root / f"eval_decision/complex_d1_{decoder}"
                directory.mkdir(parents=True)
                (directory / "summary.json").write_text(json.dumps({"status": "PASS_SYNTHETIC"}))
            for command in [[sys.executable, "/home/rna_jev/tools/compare_ripe_paired_round48.py",
                             "--root", str(root), "--manifest", "data/rna_complex_round49/d1_preparation/comparison_manifest.json",
                             "--status", "PASS_COMPLEX_D1_PAIRED_MICRO_F1_BOOTSTRAP",
                             "--out", str(root / "eval_decision/complex_d1_paired_bootstrap.json"),
                             "--bootstrap", "1000"],
                            [sys.executable, "/home/rna_jev/tools/adjudicate_complex_d1_round49.py",
                             "--root", str(root), "--out", str(root / "audits/round48/complex_d1_decision.json"),
                             "--bootstrap", "1000"]]:
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    self.fail(completed.stdout + "\n" + completed.stderr)
            decision = json.loads((root / "audits/round48/complex_d1_decision.json").read_text())
            self.assertTrue(decision["complete"])
            self.assertEqual(decision["next_base"]["experiment"], "complex_d1")
            self.assertEqual(len(decision["holdout"]), 9)
            self.assertTrue(decision["b_group_improvement"])
            self.assertFalse(decision["main_8_sota_observed"])


if __name__ == "__main__":
    unittest.main()
