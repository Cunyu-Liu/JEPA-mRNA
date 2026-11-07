import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ComplexTrainingPreparationTests(unittest.TestCase):
    def test_merge_and_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decon = root / "decon"
            decon.mkdir()
            complex_row = {"name": "X_A", "seq": "ACGUACGU", "pairs": [[1, 8]],
                           "canonical_nested_pairs": [[1, 8]], "raw_pairs": [[1, 8]]}
            (decon / "complex_train.jsonl").write_text(json.dumps(complex_row) + "\n")
            (decon / "complex_holdout.jsonl").write_text(json.dumps(complex_row) + "\n")
            (decon / "decontamination.json").write_text("{}\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            verification = {"status": "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT",
                            "decontamination_sha256": digest(decon / "decontamination.json"),
                            "train_sha256": digest(decon / "complex_train.jsonl"),
                            "holdout_sha256": digest(decon / "complex_holdout.jsonl")}
            (decon / "verification.json").write_text(json.dumps(verification))
            (root / "tr1c.jsonl").write_text('{"seq":"AAAA"}\n{"seq":"CCCC"}\n')
            for name, sequences in [("teacher_old", ["AAAA", "CCCC"]), ("teacher_new", ["ACGUACGU"])]:
                directory = root / name
                directory.mkdir()
                (directory / "shard_00000.npz").write_bytes(name.encode())
                manifest = {"teacher": "thermo:viennarna", "version_lock": "ViennaRNA 2.7.2",
                            "n_sequences": len(sequences), "shards": [{"file": "shard_00000.npz", "n": len(sequences)}]}
                (directory / "manifest.json").write_text(json.dumps(manifest))
            decision = root / "audits/round48/pairjepa_decision.json"
            decision.parent.mkdir(parents=True)
            decision.write_text(json.dumps({"complete": True, "allow_complex_data_route": True,
                                             "next_base": {"experiment": "j", "run_prefix": "j_s"}}))
            for seed in range(3):
                run = root / "runs" / f"j_s{seed}"
                run.mkdir(parents=True)
                meta = {"converged": True, "early_stopped": True, "status": "completed",
                        "best_validation_loss": 20.0 + seed,
                        "schedule": {"steps": 0, "batch_size": 2}, "validation": {"patience": 6},
                        "objective": {"lambda_jepa": 0.0}, "scorer": "pairmlp",
                        "pair_latent_dim": 490, "ripe": {"mode": "log", "encoder": False}}
                (run / "run_meta.json").write_text(json.dumps(meta))
            out = root / "out"
            command = [sys.executable, "/home/rna_jev/tools/prepare_complex_training_round49.py",
                       "--decontamination", str(decon), "--teacher-complex", str(root / "teacher_new"),
                       "--tr1c", str(root / "tr1c.jsonl"), "--teacher-tr1c", str(root / "teacher_old"),
                       "--decision", str(decision), "--out", str(out)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            config = json.loads((out / "d1_config.json").read_text())
            self.assertEqual(config["d1"]["n_complex_train"], 1)
            self.assertEqual(config["d1"]["n_tr1c"], 2)
            self.assertEqual(len((out / "train_d1.jsonl").read_text().splitlines()), 3)
            teacher = json.loads((out / "teacher/manifest.json").read_text())
            self.assertEqual(teacher["n_sequences"], 3)
            self.assertEqual(len(teacher["shards"]), 2)
            self.assertTrue(all((out / "teacher" / shard["file"]).is_file() for shard in teacher["shards"]))
            self.assertTrue(config["d0"]["reuse_existing_three_seed_results"])


if __name__ == "__main__":
    unittest.main()
