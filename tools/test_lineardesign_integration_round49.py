import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LinearDesignPipelineTests(unittest.TestCase):
    def test_frozen_preparation_and_summarization_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audits/round48"
            audit.mkdir(parents=True)
            baseline = {"experiment": "ripe_r1", "run_prefix": "ripe_head_r1_s", "mean_best_vl0": 18.0}
            (audit / "complex_d1_decision.json").write_text(json.dumps({
                "complete": True, "allow_lineardesign": True, "next_base": baseline,
                "candidates": [baseline, {"experiment": "complex_d1", "run_prefix": "complex_d1_s", "mean_best_vl0": 19.0}]}))
            for prefix in ["ripe_head_r1_s", "complex_d1_s"]:
                for seed in range(3):
                    run = root / "runs" / f"{prefix}{seed}"
                    run.mkdir(parents=True)
                    (run / "best.pt").write_bytes(f"{prefix}{seed}".encode())
                    (run / "run_meta.json").write_text(json.dumps({
                        "status": "completed", "converged": True, "early_stopped": True,
                        "best_validation_loss": 20.0 - seed, "best_step": 1000 + seed,
                        "schedule": {"steps": 0}, "validation": {"patience": 6}}))
            output = root / "eval_decision/lineardesign_round49"
            environment = dict(os.environ, PYTHONPATH=os.environ["PYTHONPATH"])
            subprocess.run([sys.executable, "/home/rna_jev/tools/prepare_lineardesign_round49.py",
                            "--root", str(root), "--lineardesign", "/home/rna_jev/third_party/LinearDesign",
                            "--out-dir", str(output)], check=True, capture_output=False, env=environment)
            manifest = json.loads((output / "manifest.json").read_text())
            rows = [json.loads(line) for line in (output / "candidates.jsonl").open()]
            self.assertEqual([row["protein"] for row in rows], ["MNDTEAI", "MPNTLACP", "MLDQVNKLKYPEVSLT*"])
            for model in manifest["models"]:
                self.assertEqual(model["seed"], 2)
                predictions = [[{"name": row["name"], "pred_pairs": row["pairs"]} for row in rows] for _ in range(3)]
                result = {"status": "PASS", "device": "cuda", "raw_n": 3, "n": 3, "repeats": 3,
                          "data_sha256": manifest["candidate_data_sha256"],
                          "checkpoint_sha256": model["checkpoint_sha256"], "checkpoint_step": model["best_step"],
                          "score_forward_count_per_sequence": 1, "prior_weight_override": None,
                          "decoder": "nussinov", "predictions": predictions, "wall_seconds": 1.0}
                (output / f"{model['experiment']}.json").write_text(json.dumps(result))
            subprocess.run([sys.executable, "/home/rna_jev/tools/summarize_lineardesign_round49.py",
                            "--root", str(root), "--out", str(output / "summary.json")],
                           check=True, capture_output=False, env=environment)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "PASS_LINEARDESIGN_MODEL_EVALUATION")
            self.assertEqual(summary["models"]["complex_d0"]["micro_f1_vs_lineardesign_mfe"], 1.0)
            self.assertFalse(summary["wet_lab_validation"])
            self.assertFalse(summary["designed_structure_ground_truth"])
            subprocess.run([sys.executable, "/home/rna_jev/tools/summarize_lineardesign_round49.py",
                            "--root", str(root), "--out", str(output / "summary.recheck.json")],
                           check=True, capture_output=False, env=environment)
            self.assertEqual(hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
                             hashlib.sha256((output / "summary.recheck.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
